"""Transformer for ASR in the SpeechBrain style.

Authors
* Jianyuan Zhong 2020
"""
from speechbrain.nnet.linear import Linear
from speechbrain.nnet.containers import ModuleList
from speechbrain.lobes.models.transformer.Transformer import (
    TransformerEncoder,
    TransformerDecoder,
    PositionalEncoding,
    get_lookahead_mask,
    get_key_padding_mask,
    NormalizedEmbedding,
)
from speechbrain.dataio.dataio import length_to_mask

import torch
import torch.nn as nn
import torch.nn.functional as F
import speechbrain as sb
from typing import Optional
import numpy as np
from utils import FiLM

from speechbrain.nnet.activations import Swish
from speechbrain.nnet.attention import RelPosEncXL

class MyTransformerEncoder(TransformerEncoder):
    def __init__(
            self,
            tgt_vocab,
            d_model=512,
            lang_dim=4,
            **kwargs
    ):
        super().__init__(d_model=d_model, **kwargs)

        self.normalize_before = True
        self.after_norm = nn.LayerNorm(d_model)

    # override
    def forward(
            self,
            src,
            src_mask: Optional[torch.Tensor] = None,
            src_key_padding_mask: Optional[torch.Tensor] = None,
            pos_embs: Optional[torch.Tensor] = None,
    ):
        output = src
        if self.layerdrop_prob > 0.0:
            keep_probs = self.rng.random(len(self.layers))
        else:
            keep_probs = None
        attention_lst = []
        intermediate_outs = []
        for i, enc_layer in enumerate(self.layers):
            if (
                    not self.training
                    or self.layerdrop_prob == 0.0
                    or keep_probs[i] > self.layerdrop_prob
            ):
                output, attention = enc_layer(
                    output,
                    src_mask=src_mask,
                    src_key_padding_mask=src_key_padding_mask,
                    pos_embs=pos_embs,
                )
                attention_lst.append(attention)
        output = self.norm(output)
        return output, attention_lst, intermediate_outs

class MyTransformerInterface(nn.Module):
    """This is an interface for transformer model.
    Adapted from speechbrain.lobes.models.transformer.Transformer.TransformerInterface
    Returns intermediate output
    """

    def __init__(
        self,
        tgt_vocab,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        d_ffn=2048,
        dropout=0.1,
        activation=nn.ReLU,
        custom_src_module=None,
        custom_tgt_module=None,
        positional_encoding="fixed_abs_sine",
        normalize_before=True,
        kernel_size: Optional[int] = 31,
        bias: Optional[bool] = True,
        encoder_module: Optional[str] = "transformer",
        conformer_activation: Optional[nn.Module] = Swish,
        branchformer_activation: Optional[nn.Module] = nn.GELU,
        attention_type: Optional[str] = "regularMHA",
        max_length: Optional[int] = 2500,
        causal: Optional[bool] = False,
        encoder_kdim: Optional[int] = None,
        encoder_vdim: Optional[int] = None,
        decoder_kdim: Optional[int] = None,
        decoder_vdim: Optional[int] = None,
        csgu_linear_units: Optional[int] = 3072,
        gate_activation: Optional[nn.Module] = nn.Identity,
        use_linear_after_conv: Optional[bool] = False,
        lang_dim=4,
    ):
        super().__init__()
        self.causal = causal
        self.attention_type = attention_type
        self.positional_encoding_type = positional_encoding
        self.encoder_kdim = encoder_kdim
        self.encoder_vdim = encoder_vdim
        self.decoder_kdim = decoder_kdim
        self.decoder_vdim = decoder_vdim

        assert attention_type in ["regularMHA", "RelPosMHAXL"]
        assert positional_encoding in ["fixed_abs_sine", None]

        assert (
            num_encoder_layers + num_decoder_layers > 0
        ), "number of encoder layers and number of decoder layers cannot both be 0!"

        if positional_encoding == "fixed_abs_sine":
            self.positional_encoding = PositionalEncoding(d_model, max_length)
        elif positional_encoding is None:
            pass
            # no positional encodings

        # overrides any other pos_embedding
        if attention_type == "RelPosMHAXL":
            self.positional_encoding = RelPosEncXL(d_model)
            self.positional_encoding_decoder = PositionalEncoding(
                d_model, max_length
            )

        # initialize the encoder
        if num_encoder_layers > 0:
            if custom_src_module is not None:
                self.custom_src_module = custom_src_module(d_model)
            assert encoder_module == "transformer"
            self.encoder = MyTransformerEncoder(
                tgt_vocab=tgt_vocab,
                # phone_vocab=phone_vocab,
                nhead=nhead,
                num_layers=num_encoder_layers,
                d_ffn=d_ffn,
                d_model=d_model,
                dropout=dropout,
                activation=activation,
                normalize_before=normalize_before,
                causal=self.causal,
                attention_type=self.attention_type,
                kdim=self.encoder_kdim,
                vdim=self.encoder_vdim,
                lang_dim=lang_dim,
            )

        # initialize the decoder
        if num_decoder_layers > 0:
            if custom_tgt_module is not None:
                self.custom_tgt_module = custom_tgt_module(d_model)
            self.decoder = TransformerDecoder(
                num_layers=num_decoder_layers,
                nhead=nhead,
                d_ffn=d_ffn,
                d_model=d_model,
                dropout=dropout,
                activation=activation,
                normalize_before=normalize_before,
                causal=True,
                attention_type="regularMHA",  # always use regular attention in decoder
                kdim=self.decoder_kdim,
                vdim=self.decoder_vdim,
            )

    def forward(self, **kwags):
        """Users should modify this function according to their own tasks."""
        raise NotImplementedError


class MyTransformerASR(MyTransformerInterface):
    """This is an implementation of transformer model for ASR.

    The architecture is based on the paper "Attention Is All You Need":
    https://arxiv.org/pdf/1706.03762.pdf

    Arguments
    ----------
    tgt_vocab: int
        Size of vocabulary.
    input_size: int
        Input feature size.
    d_model : int, optional
        Embedding dimension size.
        (default=512).
    nhead : int, optional
        The number of heads in the multi-head attention models (default=8).
    num_encoder_layers : int, optional
        The number of sub-encoder-layers in the encoder (default=6).
    num_decoder_layers : int, optional
        The number of sub-decoder-layers in the decoder (default=6).
    dim_ffn : int, optional
        The dimension of the feedforward network model (default=2048).
    dropout : int, optional
        The dropout value (default=0.1).
    activation : torch.nn.Module, optional
        The activation function of FFN layers.
        Recommended: relu or gelu (default=relu).
    positional_encoding: str, optional
        Type of positional encoding used. e.g. 'fixed_abs_sine' for fixed absolute positional encodings.
    normalize_before: bool, optional
        Whether normalization should be applied before or after MHA or FFN in Transformer layers.
        Defaults to True as this was shown to lead to better performance and training stability.
    kernel_size: int, optional
        Kernel size in convolutional layers when Conformer is used.
    bias: bool, optional
        Whether to use bias in Conformer convolutional layers.
    encoder_module: str, optional
        Choose between Conformer and Transformer for the encoder. The decoder is fixed to be a Transformer.
    conformer_activation: torch.nn.Module, optional
        Activation module used after Conformer convolutional layers. E.g. Swish, ReLU etc. it has to be a torch Module.
    branchformer_activation: torch.nn.Module, optional
        Activation module used within the Branchformer Encoder. E.g. Swish, ReLU etc. it has to be a torch Module.
    attention_type: str, optional
        Type of attention layer used in all Transformer or Conformer layers.
        e.g. regularMHA or RelPosMHA.
    max_length: int, optional
        Max length for the target and source sequence in input.
        Used for positional encodings.
    causal: bool, optional
        Whether the encoder should be causal or not (the decoder is always causal).
        If causal the Conformer convolutional layer is causal.
    csgu_linear_units: int, optional
        Number of neurons in the hidden linear units of the CSGU Module.
        -> Branchformer
    gate_activation: torch.nn.Module, optional
        Activation function used at the gate of the CSGU module.
        -> Branchformer
    use_linear_after_conv: bool, optional
        If True, will apply a linear transformation of size input_size//2.
        -> Branchformer

    Example
    -------
    >>> src = torch.rand([8, 120, 512])
    >>> tgt = torch.randint(0, 720, [8, 120])
    >>> net = TransformerASR(
    ...     720, 512, 512, 8, 1, 1, 1024, activation=torch.nn.GELU
    ... )
    >>> enc_out, dec_out = net.forward(src, tgt)
    >>> enc_out.shape
    torch.Size([8, 120, 512])
    >>> dec_out.shape
    torch.Size([8, 120, 512])
    """

    def __init__(
        self,
        tgt_vocab,
        input_size,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        d_ffn=2048,
        dropout=0.1,
        activation=nn.ReLU,
        positional_encoding="fixed_abs_sine",
        normalize_before=False,
        kernel_size: Optional[int] = 31,
        bias: Optional[bool] = True,
        encoder_module: Optional[str] = "transformer",
        conformer_activation: Optional[nn.Module] = Swish,
        branchformer_activation: Optional[nn.Module] = nn.GELU,
        attention_type: Optional[str] = "regularMHA",
        max_length: Optional[int] = 2500,
        causal: Optional[bool] = True,
        csgu_linear_units: Optional[int] = 3072,
        gate_activation: Optional[nn.Module] = nn.Identity,
        use_linear_after_conv: Optional[bool] = False,
        lang_dim=-1,
    ):
        super().__init__(
            tgt_vocab=tgt_vocab,
            # phone_vocab=phone_vocab,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            d_ffn=d_ffn,
            dropout=dropout,
            activation=activation,
            positional_encoding=positional_encoding,
            normalize_before=normalize_before,
            kernel_size=kernel_size,
            bias=bias,
            encoder_module=encoder_module,
            conformer_activation=conformer_activation,
            branchformer_activation=branchformer_activation,
            attention_type=attention_type,
            max_length=max_length,
            causal=causal,
            csgu_linear_units=csgu_linear_units,
            gate_activation=gate_activation,
            use_linear_after_conv=use_linear_after_conv,
            lang_dim=lang_dim,
        )

        self.custom_src_module = ModuleList(
            Linear(
                input_size=input_size,
                n_neurons=d_model,
                bias=True,
                combine_dims=False,
            ),
            torch.nn.Dropout(dropout),
        )
        self.custom_tgt_module = ModuleList(
            NormalizedEmbedding(d_model, tgt_vocab)
        )

        if lang_dim != -1:
            self.film = FiLM(input_size=d_model, conditioning_size=lang_dim)
        else:
            self.film = None

        # reset parameters using xavier_normal_
        self._init_params()

    def forward(self, src, tgt, wav_len=None, pad_idx=0, lang_conditioning=None, film_switch=True):
        """
        Arguments
        ----------
        src : torch.Tensor
            The sequence to the encoder.
        tgt : torch.Tensor
            The sequence to the decoder.
        wav_len: torch.Tensor, optional
            Torch Tensor of shape (batch, ) containing the relative length to padded length for each example.
        pad_idx : int, optional
            The index for <pad> token (default=0).
        """

        # reshpae the src vector to [Batch, Time, Fea] is a 4d vector is given
        if src.ndim == 4:
            bz, t, ch1, ch2 = src.shape
            src = src.reshape(bz, t, ch1 * ch2)

        (
            src_key_padding_mask,
            tgt_key_padding_mask,
            src_mask,
            tgt_mask,
        ) = self.make_masks(src, tgt, wav_len, pad_idx=pad_idx)

        src = self.custom_src_module(src)
        # add pos encoding to queries if are sinusoidal ones else
        assert self.positional_encoding_type == "fixed_abs_sine"
        src = src + self.positional_encoding(src)  # add the encodings here
        pos_embs_encoder = None

        encoder_out, _, intermediate_outs = self.encoder(
            src=src,
            src_mask=src_mask,
            src_key_padding_mask=src_key_padding_mask,
            pos_embs=pos_embs_encoder,
        )

        if self.film is not None:
            encoder_out_1 = self.film(encoder_out, lang_conditioning, film_switch)
        else:
            encoder_out_1 = encoder_out

        tgt = self.custom_tgt_module(tgt)

        # Add positional encoding to the target before feeding the decoder.
        tgt = tgt + self.positional_encoding(tgt)
        pos_embs_target = None
        pos_embs_encoder = None

        decoder_out, _, _ = self.decoder(
            tgt=tgt,
            memory=encoder_out_1,
            memory_mask=src_mask,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
            pos_embs_tgt=pos_embs_target,
            pos_embs_src=pos_embs_encoder,
        )

        return encoder_out, encoder_out_1, decoder_out

    def make_masks(self, src, tgt=None, wav_len=None, pad_idx=0):
        """This method generates the masks for training the transformer model.

        Arguments
        ---------
        src : tensor
            The sequence to the encoder (required).
        tgt : tensor
            The sequence to the decoder.
        pad_idx : int
            The index for <pad> token (default=0).
        """
        src_key_padding_mask = None
        if wav_len is not None:
            abs_len = torch.round(wav_len * src.shape[1])
            src_key_padding_mask = ~length_to_mask(abs_len).bool()

        src_mask = None

        # If no decoder in the transformer...
        if tgt is not None:
            tgt_key_padding_mask = get_key_padding_mask(tgt, pad_idx=pad_idx)
            tgt_mask = get_lookahead_mask(tgt)
        else:
            tgt_key_padding_mask = None
            tgt_mask = None

        return src_key_padding_mask, tgt_key_padding_mask, src_mask, tgt_mask

    @torch.no_grad()
    def decode(self, tgt, encoder_out, enc_len=None):
        """This method implements a decoding step for the transformer model.

        Arguments
        ---------
        tgt : torch.Tensor
            The sequence to the decoder.
        encoder_out : torch.Tensor
            Hidden output of the encoder.
        enc_len : torch.LongTensor
            The actual length of encoder states.
        """
        tgt_mask = get_lookahead_mask(tgt)
        src_key_padding_mask = None
        if enc_len is not None:
            src_key_padding_mask = (1 - length_to_mask(enc_len)).bool()

        tgt = self.custom_tgt_module(tgt)
        if self.attention_type == "RelPosMHAXL":
            # use standard sinusoidal pos encoding in decoder
            tgt = tgt + self.positional_encoding_decoder(tgt)
            pos_embs_encoder = None  # self.positional_encoding(src)
            pos_embs_target = None
        elif self.positional_encoding_type == "fixed_abs_sine":
            tgt = tgt + self.positional_encoding(tgt)
            pos_embs_target = None
            pos_embs_encoder = None

        prediction, self_attns, multihead_attns = self.decoder(
            tgt,
            encoder_out,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=src_key_padding_mask,
            pos_embs_tgt=pos_embs_target,
            pos_embs_src=pos_embs_encoder,
        )
        return prediction, multihead_attns[-1]

    def encode(self, src, wav_len=None, pad_idx=0):
        """
        Encoder forward pass

        Arguments
        ----------
        src : torch.Tensor
            The sequence to the encoder.
        wav_len: torch.Tensor, optional
            Torch Tensor of shape (batch, ) containing the relative length to padded length for each example.
        """
        # reshape the src vector to [Batch, Time, Fea] if a 4d vector is given
        if src.dim() == 4:
            bz, t, ch1, ch2 = src.shape
            src = src.reshape(bz, t, ch1 * ch2)

        (src_key_padding_mask, _, src_mask, _,) = self.make_masks(
            src, None, wav_len, pad_idx=pad_idx
        )

        src = self.custom_src_module(src)
        if self.attention_type == "RelPosMHAXL":
            pos_embs_source = self.positional_encoding(src)

        elif self.positional_encoding_type == "fixed_abs_sine":
            src = src + self.positional_encoding(src)
            pos_embs_source = None

        encoder_out, _ = self.encoder(
            src=src,
            src_key_padding_mask=src_key_padding_mask,
            pos_embs=pos_embs_source,
        )
        return encoder_out

    def _init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_normal_(p)
