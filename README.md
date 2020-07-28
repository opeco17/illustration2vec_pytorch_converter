## How to convert caffe model to pytorch model

```
from convert import Converter

Converter.convert_feature_params('../illust2vec_ver200.caffemodel')
Converter.convert_tag_params('../illust2vec_tag_ver200.caffemodel')
```

## How to extract feature and tags

```
from extract import Extractor

feature = Extractor.feature_extract('feature_parameter.pth', ['images/test.png'])
tag = Extractor.tag_extract('tag_parameter.pth', ['images/test.png'])
```
