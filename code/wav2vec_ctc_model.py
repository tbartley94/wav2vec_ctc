
from typing import Dict, Optional, List, Tuple
from nemo.core.neural_types.elements import Target

import torch
from torch import nn
from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict
from collections import OrderedDict
from nemo.core.classes.common import PretrainedModelInfo, typecheck

from wav2vec.wav2vec_modules import compute_mask_indices, GumbelVectorQuantizer
from nemo.collections.asr.models.ctc_bpe_models import EncDecCTCModelBPE
from wav2vec.wav2vec import GradMultiply
from nemo.collections.asr.parts.submodules.jasper import (
    init_weights,
)
from nemo.core.classes.common import typecheck
from nemo.core.classes.exportable import Exportable
from nemo.core.classes.module import NeuralModule
from nemo.core.neural_types import (
    AcousticEncodedRepresentation,
    LengthsType,
    AudioSignal,
    LogprobsType,
    NeuralType,
    SpectrogramType,
)
from torch.nn import functional as F
"""
Class for configuring wav2vec-ctc. Default wav2vec encoder has different outputs than
assumed by ctc decoder. So this class alters the forward pass to allow passing of weights.

Does the following:
-- Takes a config file for ctc with wav2vec encoder
-- sets encoder to remove pretraining features (see wav2vec_models file)
-- Takes outputs of encoder pass and makes them appropriate for decoder
-- feeds to decoder
-- uses defaults from EncDecCTCModelBPE

Eventually should take a wav2vec encoder config and produce a CTC-BPE enc-dec model.

"""


class Wav2VecCTCModel(EncDecCTCModelBPE):
    @classmethod
    def list_available_models(cls) -> Optional[PretrainedModelInfo]:
        return None

    def __init__(self, cfg: DictConfig, trainer = None):
        # Let CTCBPE manage tokenization and encoder-decoder assignment
        super().__init__(cfg=cfg, trainer=trainer)

        # TODO: Make it that it manages pretrained modules better
        # Also provide the option to put a new decoder on top
        # include freezing of convolutions given pretrained model

        # stuff we take from wav2vec_model
        # embeddings from feature extractor

        # configs for extractor and preprocessing
        self.embed = cfg.preprocessor.conv_layers[-1][0]  # Select last conv output layer dimension
        self.layers = cfg.preprocessor.conv_layers

        self.normalize_audio = cfg.normalize_audio

        self.feature_grad_mult = cfg.feature_grad_mult

        self.feature_loss_weight = cfg.feature_loss_weight
        self.features_penalty = 0.0

        # configs for main model
        encoder_embed_dim = cfg.final_dim
        self.post_extract_proj = (
            nn.Linear(self.embed, encoder_embed_dim)
            if self.embed != encoder_embed_dim and not cfg.quantizer.quantize_input
            else None
        )
        final_dim = cfg.final_dim if cfg.final_dim > 0 else encoder_embed_dim
        self.final_dim = final_dim


        # Masking 
        self.mask_cfg = cfg.masking 
        self.mask_emb = nn.Parameter(torch.FloatTensor(encoder_embed_dim).uniform_())

        # Dropout
        self.dropout_input = nn.Dropout(cfg.dropout_input)

        # Quantization information
        self.input_quantizer = None

        #Leaving in quantizer incase it's desired
        if cfg.quantizer.quantize_input:
            vq_dim = cfg.quantizer.latent_dim if cfg.quantizer.latent_dim > 0 else encoder_embed_dim
            self.input_quantizer = GumbelVectorQuantizer(
                dim=self.embed,
                num_vars=cfg.quantizer.latent_vars,
                temp=cfg.quantizer.latent_temp,
                groups=cfg.quantizer.latent_groups,
                combine_groups=False,
                vq_dim=vq_dim,
                time_first=True,
            )
            self.project_inp = nn.Linear(vq_dim, encoder_embed_dim)


        self.layer_norm = nn.LayerNorm(self.embed)

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        if hasattr(self.preprocessor, '_sample_rate'):
            input_signal_eltype = AudioSignal(freq=self.preprocessor._sample_rate)
        else:
            input_signal_eltype = AudioSignal()
        return {
            "input_signal": NeuralType(('B', 'T'), input_signal_eltype, optional=True),
            "input_signal_length": NeuralType(tuple('B'), LengthsType(), optional=True),
            "processed_signal": NeuralType(('B', 'T'), SpectrogramType(), optional=True),
            "processed_signal_length": NeuralType(tuple('B'), LengthsType(), optional=True),
            "sample_id": NeuralType(tuple('B'), LengthsType(), optional=True),
        }

    @typecheck()
    def forward(
        self, input_signal=None, input_signal_length=None, processed_signal=None, processed_signal_length=None
    ):
        """
        Forward pass of the model.

        Args:
            input_signal: Tensor that represents a batch of raw audio signals,
                of shape [B, T]. T here represents timesteps, with 1 second of audio represented as
                `self.sample_rate` number of floating point values.
            input_signal_length: Vector of length B, that contains the individual lengths of the audio
                sequences.
            processed_signal: Tensor that represents a batch of processed audio signals,
                of shape (B, T) that has undergone processing via some DALI preprocessor.
            processed_signal_length: Vector of length B, that contains the individual lengths of the
                processed audio sequences.

        Returns:
            A tuple of 3 elements -
            1) The log probabilities tensor of shape [B, T, D].
            2) The lengths of the acoustic sequence after propagation through the encoder, of shape [B].
            3) The greedy token predictions of the model of shape [B, T] (via argmax)
        """
        has_input_signal = input_signal is not None and input_signal_length is not None
        has_processed_signal = processed_signal is not None and processed_signal_length is not None
        if (has_input_signal ^ has_processed_signal) == False:
            raise ValueError(
                f"{self} Arguments ``input_signal`` and ``input_signal_length`` are mutually exclusive "
                " with ``processed_signal`` and ``processed_signal_len`` arguments."
            )
        
        # Remove processing if we're given raw audio

        if has_input_signal:
            signal = input_signal
            audio_length = input_signal_length
            mask = False
        else:
            signal = processed_signal
            audio_length = processed_signal_length
            mask = True
        if self.normalize_audio:
            with torch.no_grad():
                signal = normalize(signal, audio_length)
        padding = self._create_padding_mask(audio_length)
        encoded_length = self.get_lengths(audio_lengths=audio_length)
        features, padding = self._get_features(signal, padding_mask=padding, mask=mask)
        logits = self.encoder(features, padding_mask=padding)
        log_probs = self.decoder(encoder_output=logits)
        greedy_predictions = log_probs.argmax(dim=-1, keepdim=False)
        return log_probs, encoded_length, greedy_predictions
    
    # Calls convolutional layer to get feature embeddings for audio
    def _get_features(self, source, padding_mask=None, mask=False):
        if self.feature_grad_mult > 0:
            features = self.preprocessor(source)
            if self.feature_grad_mult != 1.0:
                features = GradMultiply.apply(features, self.feature_grad_mult)
        else:
            with torch.no_grad():
                features = self.preprocessor(source)

        if self.feature_loss_weight: # Store as instance variable so we don't have to pass through forward
            self.features_penalty = features.float().pow(2).mean()

        features = features.transpose(1, 2)
        features = self.layer_norm(features)

        if padding_mask is not None:
            extra = padding_mask.size(1) % features.size(1)
            if extra > 0:
                padding_mask = padding_mask[:, :-extra]
            padding_mask = padding_mask.view(padding_mask.size(0), features.size(1), -1)
            padding_mask = padding_mask.all(-1)

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        features = self.dropout_input(features)

        if self.input_quantizer:
            self._update_quantizer_temp()
            features, _, _ = self.input_quantizer(features)
            features = self.project_inp(features)

        if mask:
            features, _ = self.apply_mask(features, padding_mask)

        return features, padding_mask
    

    # PTL-specific methods
    def training_step(self, batch, batch_nb):
        signal, signal_len, transcript, transcript_len = batch

        # Come back to later if dali is important
        # if isinstance(batch, DALIOutputs) and batch.has_processed_signal:
        #     log_probs, encoded_len, predictions = self.forward(
        #         processed_signal=signal, processed_signal_length=signal_len
        #     )
        # else:
        #     log_probs, encoded_len, predictions = self.forward(input_signal=signal, input_signal_length=signal_len)
        log_probs, encoded_len, predictions  = self.forward(processed_signal=signal, processed_signal_length=signal_len)
        loss_value = self.loss(
            log_probs=log_probs, targets=transcript, input_lengths=encoded_len, target_lengths=transcript_len
        )
        if self.features_penalty:
            loss_value += self.features_penalty
            self.features_penalty = 0.0 # reset

        current_hypotheses = self._wer.ctc_decoder_predictions_tensor(
                            predictions, predictions_len=encoded_len, return_hypotheses=False,
                        )
        print(current_hypotheses)
        tensorboard_logs = {'train_loss': loss_value, 'learning_rate': self._optimizer.param_groups[0]['lr']}
        if hasattr(self, '_trainer') and self._trainer is not None:
            log_every_n_steps = self._trainer.log_every_n_steps
        else:
            log_every_n_steps = 1

        if (batch_nb + 1) % log_every_n_steps == 0:
            self._wer.update(
                predictions=predictions,
                targets=transcript,
                target_lengths=transcript_len,
                predictions_lengths=encoded_len,
            )
            wer, _, _ = self._wer.compute()
            self._wer.reset()
            tensorboard_logs.update({'training_batch_wer': wer})
        return {'loss': loss_value, 'log': tensorboard_logs}


    def _create_padding_mask(self, audio_lengths):
        # Broadcast to vectorize creating the padding mask
        max_len = max(audio_lengths)
        padding_mask = torch.arange(max_len, device=self.device)
        padding_mask = padding_mask.expand(len(audio_lengths), max_len) < audio_lengths.unsqueeze(1)
        # Negate to false where no padding
        padding_mask = ~padding_mask
        return padding_mask


    def apply_mask(self, x, padding_mask):
        B, T, C = x.shape
        if self.mask_cfg.mask_prob > 0:
            mask_indices = compute_mask_indices(
                (B, T),
                padding_mask,
                self.mask_cfg.mask_prob,
                self.mask_cfg.mask_length,
                self.mask_cfg.mask_type,
                mask_other= self.mask_cfg.mask_other,
                min_masks=2,
                no_overlap=self.mask_cfg.no_mask_overlap,
                min_space=self.mask_cfg.mask_min_space,
            )
            mask_indices = torch.from_numpy(mask_indices).to(x.device)
            mask_emb = self.mask_emb.type_as(x)
            x[mask_indices] = mask_emb
        else:
            mask_indices = None

        if self.mask_cfg.mask_channel_prob > 0:
            mask_channel_indices = compute_mask_indices(
                (B, C),
                None,
                self.mask_cfg.mask_channel_prob,
                self.mask_cfg.mask_channel_length,
                self.mask_cfg.mask_channel_type,
                mask_other=self.mask_cfg.mask_channel_other,
                min_masks=2,
                no_overlap=self.mask_cfg.no_mask_channel_overlap,
                min_space=self.mask_cfg.mask_channel_min_space,
            )
            mask_channel_indices = torch.from_numpy(mask_channel_indices).to(x.device).unsqueeze(1).expand(-1, T, -1)
            x[mask_channel_indices] = 0

        return x, mask_indices

    def get_lengths(self, audio_lengths):
        for conv in self.layers:
            kernel = conv[1]
            stride = conv[2]
            audio_lengths = torch.div(audio_lengths - kernel, stride, rounding_mode='floor')
        return audio_lengths

    def _update_quantizer_temp(self):
        if self.input_quantizer:
            self.input_quantizer.set_num_updates(self.trainer.global_step)

class Wav2VecLinearDecoder(NeuralModule, Exportable):
    """Simple ASR Decoder for linear projection of Wav2Vec embeddings
    """

    def save_to(self, save_path: str):
        pass

    @classmethod
    def restore_from(cls, restore_path: str):
        pass

    @property
    def input_types(self):
        return OrderedDict({"encoder_output": NeuralType(('B', 'T', 'D'), AcousticEncodedRepresentation())})

    @property
    def output_types(self):
        return OrderedDict({"logprobs": NeuralType(('B', 'T', 'D'), LogprobsType())})

    def __init__(self, feat_in, num_classes, init_mode="xavier_uniform", vocabulary=None):
        super().__init__()

        if vocabulary is not None:
            if num_classes != len(vocabulary):
                raise ValueError(
                    f"If vocabulary is specified, it's length should be equal to the num_classes. Instead got: num_classes={num_classes} and len(vocabulary)={len(vocabulary)}"
                )
            self.__vocabulary = vocabulary
        self._feat_in = feat_in
        # Add 1 for blank char
        self._num_classes = num_classes + 1

        self.projection = torch.nn.Linear(self._feat_in, self._num_classes, bias=False)
        self.apply(lambda x: init_weights(x, mode=init_mode))

    @typecheck()
    def forward(self, encoder_output):
        return torch.nn.functional.log_softmax(self.projection(encoder_output), dim=-1)

    def input_example(self):
        """
        Generates input examples for tracing etc.
        Returns:
            A tuple of input examples.
        """
        bs = 8
        seq = 64
        input_example = torch.randn(bs, self._feat_in, seq).to(next(self.parameters()).device)
        return tuple([input_example])

    def _prepare_for_export(self, **kwargs):
        pass

    @property
    def vocabulary(self):
        return self.__vocabulary

    @property
    def num_classes_with_blank(self):
        return self._num_classes

class ConvFeatureEncoder(nn.Module):
    """
        Converts input raw audio into features for downstream transformer model.
        Uses 1D convolutional blocks with GeLU activation.
    """

    def __init__(
        self,
        conv_layers: List[Tuple[int, int, int]],
        mode: str,
        conv_bias: bool = False,
    ):
        super().__init__()
        self.mode = mode

        def block(
            n_in, n_out, k, stride, is_layer_norm=False, is_group_norm=False, conv_bias=False,
        ):
            def make_conv():
                conv = nn.Conv1d(n_in, n_out, k, stride=stride, bias=conv_bias)
                nn.init.kaiming_normal_(conv.weight)
                return conv

            assert (is_layer_norm and is_group_norm) is False, "layer norm and group norm are exclusive"
            if is_layer_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Sequential(TransposeLast(), nn.LayerNorm(dim, elementwise_affine=True), TransposeLast()),
                    nn.GELU(),
                )
            elif is_group_norm:
                return nn.Sequential(make_conv(), nn.GroupNorm(dim, dim, affine=True), nn.GELU(),)
            else:
                return nn.Sequential(make_conv(), nn.GELU())

        in_d = 1
        self.conv_layers = nn.ModuleList()
        for i, cl in enumerate(conv_layers):
            assert len(cl) == 3, "invalid conv definition: " + str(cl)
            (dim, k, stride) = cl

            self.conv_layers.append(
                block(
                    in_d,
                    dim,
                    k,
                    stride,
                    is_layer_norm=mode == "layer_norm",
                    is_group_norm=mode == "group_norm" and i == 0,
                    conv_bias=conv_bias,
                )
            )
            in_d = dim

    def forward(self, x):
        # BxT -> BxCxT
        x = x.unsqueeze(1)
        for conv in self.conv_layers:
            x = conv(x)
        return x

class TransposeLast(torch.nn.Module):
    """
    Transposes last dimension. Useful for adding to a sequential block.
    """

    def forward(self, x):
        return x.transpose(-2, -1)

def normalize(source, lengths):
    for i in range(lengths.size(0)):
        orig = source[i, :lengths[i]]
        norm = F.layer_norm(orig, orig.shape) # From FAIR
        source[i, :lengths[i]] = norm
    return source


