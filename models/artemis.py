import gc
import math
import os
import shutil
import zipfile
from typing import List, Union
from urllib.request import urlretrieve

import numpy as np
import torch
import torch.nn.functional as F
import torchvision as tv
from PIL.Image import Image

from artemis.in_out.neural_net_oriented import load_saved_speaker, torch_load_model
from models.model import Model

speaker_saved_args = "cache/modelzoo/artemis/config.json.txt"
speaker_checkpoint = "cache/modelzoo/artemis/speaker.pt"
img2emo_checkpoint = "cache/modelzoo/artemis/img2emo.pt"
os.makedirs("cache/modelzoo/artemis", exist_ok=True)


class ArtemisSubModule(torch.nn.Module):
    def __init__(self, caption, emotion, device):
        super().__init__()
        self.caption = caption.to(device)

        self.emotion = emotion
        for name, mod in self.emotion.img_encoder.named_modules():
            if isinstance(mod, torch.nn.Conv2d):
                mod._reversed_padding_repeated_twice = [*mod.padding, *mod.padding]
        self.emotion = self.emotion.to(device)

        self.device = device

    def beam_search(self, image, beam_size=5, max_iter=240, temperature=0.5, drop_unk=True, drop_bigrams=True):
        """
        Adapted from artemis.neural_models.attentive_decoder.sample_captions_beam_search
        """
        image = image.to(self.device)

        aux_data = self.emotion(image)
        aux_feat = self.caption.decoder.auxiliary_net(aux_data)

        k = beam_size
        encoder_out = self.caption.encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
        enc_image_size = encoder_out.size(1)
        encoder_dim = encoder_out.size(3)

        # Flatten encoding
        encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)

        # We'll treat the problem as having a batch size of k
        encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)

        # Tensor to store top k previous words at each step; now they're just <sos>
        k_prev_words = torch.LongTensor([[self.caption.decoder.vocab.sos]] * k).to(self.device)  # (k, 1)

        # Tensor to store top k sequences; now they're just <sos>
        seqs = k_prev_words  # (k, 1)

        # Tensor to store top k sequences' scores; now they're just 0
        top_k_scores = torch.zeros(k, 1).to(self.device)  # (k, 1)

        # Tensor to store top k sequences' alphas; now they're just 1s
        seqs_alpha = torch.ones(k, 1, enc_image_size, enc_image_size).to(
            self.device
        )  # (k, 1, enc_image_size, enc_image_size)

        # Lists to store completed sequences and scores
        complete_seqs = list()
        complete_seqs_scores = list()

        # Start decoding
        step = 1
        h, c = self.caption.decoder.init_hidden_state(encoder_out)

        # s (below) is a number less than or equal to k, sequences are removed from this process once they hit <eos>
        while True:
            embeddings = self.caption.decoder.word_embedding(k_prev_words).squeeze(1)  # (s, embed_dim)
            awe, alpha = self.caption.decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)
            alpha = alpha.view(-1, enc_image_size, enc_image_size)  # (s, enc_image_size, enc_image_size)
            gate = self.caption.decoder.sigmoid(self.caption.decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
            awe = gate * awe
            decoder_input = torch.cat([embeddings, awe], dim=1)

            af = torch.repeat_interleave(aux_feat, decoder_input.shape[0], dim=0)
            decoder_input = torch.cat([decoder_input, af], dim=1)

            h, c = self.caption.decoder.decode_step(decoder_input, (h, c))  # (s, decoder_dim)
            scores = self.caption.decoder.next_word(h)  # (s, vocab_size)

            if temperature != 1:
                scores /= temperature

            scores = F.log_softmax(scores, dim=1)

            if drop_unk:
                scores[:, self.caption.decoder.vocab.unk] = -math.inf

            if drop_bigrams and step > 2:
                # drop bi-grams with frequency higher than 1.
                prev_usage = seqs[:, : step - 1]
                x, y = torch.where(prev_usage == k_prev_words)
                y += 1  # word-after-last-in-prev-usage
                y = seqs[x, y]
                scores[x, y] = -math.inf

            if step > 2:
                ## drop x and x
                and_token = self.caption.decoder.vocab("and")
                x, y = torch.where(k_prev_words == and_token)
                pre_and_word = seqs[x, step - 2]
                scores[x, pre_and_word] = -math.inf

            # Add log-probabilities
            scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

            # For the first step, all k points will have the same scores (since same k previous words, h, c)
            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
            else:
                # Unroll and find top scores, and their unrolled indices
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

            # Convert unrolled indices to actual indices of scores
            prev_word_inds = (top_k_words / len(self.caption.decoder.vocab)).long()  # (s)
            next_word_inds = (top_k_words % len(self.caption.decoder.vocab)).long()  # (s)

            # Add new words to sequences
            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)
            seqs_alpha = torch.cat(
                [seqs_alpha[prev_word_inds], alpha[prev_word_inds].unsqueeze(1)], dim=1
            )  # (s, step+1, enc_image_size, enc_image_size)

            # Which sequences are incomplete (didn't reach <eos>)?
            incomplete_inds = []
            for ind, word in enumerate(next_word_inds):
                if word != self.caption.decoder.vocab.eos:
                    incomplete_inds.append(ind)
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

            # Set aside complete sequences
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].detach().tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds].detach().tolist())
            k -= len(complete_inds)  # reduce beam length accordingly

            # Proceed with incomplete sequences
            if k == 0:
                break
            seqs = seqs[incomplete_inds]
            seqs_alpha = seqs_alpha[incomplete_inds]

            h = h[prev_word_inds[incomplete_inds]]
            c = c[prev_word_inds[incomplete_inds]]
            encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

            # Break if things have been going on too long
            if step > max_iter:
                break
            step += 1

        s_idx = np.argsort(complete_seqs_scores)[::-1]
        best_seq = complete_seqs[s_idx[0]]
        caption = self.caption.decoder.vocab.decode_print(best_seq)
        return caption

    def forward(self, image):
        return self.beam_search(image)


class Artemis(Model):
    def __init__(self):
        self.model = None
        self.output_size = -1

        if not os.path.exists(img2emo_checkpoint):
            print("Downloading image to emotion classifier...")
            urlretrieve("https://www.dropbox.com/s/8dfj3b36q15iieo/best_model.pt?dl=1", img2emo_checkpoint)

        if not os.path.exists(speaker_checkpoint):
            print("Downloading emotion-grounded speaker...")
            path, _ = urlretrieve("https://www.dropbox.com/s/0erh464wag8ods1/emo_grounded_sat_speaker_cvpr21.zip?dl=1")
            with zipfile.ZipFile(path, "r") as f:
                f.extractall("cache/")
            shutil.move("cache/03-17-2021-20-32-19/checkpoints/best_model.pt", speaker_checkpoint)
            shutil.move("cache/03-17-2021-20-32-19/config.json.txt", speaker_saved_args)
            shutil.rmtree("cache/03-17-2021-20-32-19")

        self.normalize = tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def initialize(self, device):
        caption_model, _, _ = load_saved_speaker(
            speaker_saved_args, speaker_checkpoint, override_args=dict(data_dir="cache/modelzoo/artemis")
        )
        caption_model.eval()

        emotion_classifier = torch.load(img2emo_checkpoint, map_location="cpu")

        self.device = device
        self.model = ArtemisSubModule(caption_model, emotion_classifier, device)

    def __call__(self, inputs: List[Union[Image, torch.Tensor]]):
        if not isinstance(inputs, list):
            inputs = [inputs]
        outputs = []
        for img_or_tensor in inputs:
            if isinstance(img_or_tensor, Image):
                img_or_tensor = tv.transforms.ToTensor()(img_or_tensor)
            tensor = F.interpolate(img_or_tensor, size=(256, 256), mode="bicubic", align_corners=False)
            tensor = self.normalize(tensor)
            outputs.append(self.model(tensor))
        return outputs
