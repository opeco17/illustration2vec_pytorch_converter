import json
import numpy as np
import torch
from PIL import Image

from model import TagI2V, FeatureI2V


class Extractor(object):

    def feature_extract(self, pytorch_parameter_path, img_paths):
        i2v = FeatureI2V()
        i2v.load_state_dict(torch.load(pytorch_parameter_path))

        imgs = [Image.open(img_path) for img_path in img_paths]
        img_tensors = [self._img_to_tensor(img) for img in imgs]
        img_tensors = torch.stack(img_tensors, axis=0)
        features = i2v(img_tensors).detach().numpy()
        result = [features[i].reshape(-1) for i in range(features.shape[0])]
        return result
    

    def tag_extract(self, pytorch_parameter_path, tag_path, img_paths, n_tag=10):
        i2v = TagI2V()
        i2v.load_state_dict(torch.load(pytorch_parameter_path))

        imgs = [Image.open(img_path) for img_path in img_paths]
        img_tensors = [self._img_to_tensor(img) for img in imgs]
        img_tensors = torch.stack(img_tensors, axis=0)

        prob = i2v(img_tensors).detach().numpy()

        general_prob = prob[:, :512]
        character_prob = prob[:, 512:1024]
        copyright_prob = prob[:, 1024:1536]
        rating_prob = prob[:, 1536:]

        general_arg = np.argsort(-general_prob, axis=1)[:, :n_tag]
        character_arg = np.argsort(-character_prob, axis=1)[:, :n_tag]
        copyright_arg = np.argsort(-copyright_prob, axis=1)[:, :n_tag]
        rating_arg = np.argsort(-rating_prob, axis=1)

        tags = np.array(json.loads(open(tag_path, 'r').read()))
        result = []

        for i in range(prob.shape[0]):
            result.append({
                'general': list(zip(
                    tags[general_arg[i]],
                    general_prob[i, general_arg[i]].tolist())),
                'character': list(zip(
                    tags[512 + character_arg[i]],
                    character_prob[i, character_arg[i]].tolist())),
                'copyright': list(zip(
                    tags[1024 + copyright_arg[i]],
                    copyright_prob[i, copyright_arg[i]].tolist())),
                'rating': list(zip(
                    tags[1536 + rating_arg[i]],
                    rating_prob[i, rating_arg[i]].tolist())),
            })
        
        return result


    def _img_to_tensor(self, img):
        img = img.convert('RGB')
        mean = np.array([164.76139251,  167.47864617,  181.13838569])
        img_npy = np.array(img.resize((224, 224)), dtype='float32')
        if img_npy.shape[2] == 4:
            img_npy = img_npy[:, :, :3]
        if np.ndim(img_npy) == 2:
            img_npy = np.stack((img_npy, )*3, axis=-1)
        img_npy = img_npy.reshape(224, 224, 3)
        img_npy = img_npy[:, :, ::-1]
        img_npy -= mean
        img_npy = img_npy.transpose(2, 0, 1)
        img_tensor = torch.from_numpy(img_npy.copy())
        return img_tensor