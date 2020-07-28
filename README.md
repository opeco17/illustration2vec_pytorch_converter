# illustration2vec_PyTorch
This is the pytorch implementation of [illustration2vec](https://github.com/rezoo/illustration2vec).

To use parameters of model, you should download caffe parameters from [here](https://github.com/rezoo/illustration2vec/releases).

Converter module converts these parameters to PyTorch ones.

## How to convert caffe parameters to pytorch ones

```
from convert import Converter

Converter.convert_feature_params('../illust2vec_ver200.caffemodel')
Converter.convert_tag_params('../illust2vec_tag_ver200.caffemodel')
```

## How to extract feature and tags

```
from extract import Extractor

feature = Extractor.feature_extract('feature_parameter.pth', 'tag_list.json', ['images/test.png'])
tag = Extractor.tag_extract('tag_parameter.pth', ['images/test.png'])
```

# Example
