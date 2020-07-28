# illustration2vec_PyTorch
This is the pytorch implementation of [illustration2vec](https://github.com/rezoo/illustration2vec).

To use parameters of model, you should download caffe parameters from [here](https://github.com/rezoo/illustration2vec/releases).

Converter module converts these parameters to PyTorch ones.

## How to convert caffe parameters to pytorch ones

```
from convert import Converter

Converter.convert_feature_params('illust2vec_ver200.caffemodel')
Converter.convert_tag_params('illust2vec_tag_ver200.caffemodel')
```

## How to extract feature and tags

```
from extract import Extractor

feature = Extractor.feature_extract('feature_parameter.pth', 'tag_list.json', ['images/test.png'])
tag = Extractor.tag_extract('tag_parameter.pth', ['images/test.png'])
```

## Example
```
{'general': [
('gloves', 0.9831963777542114), 
('elbow gloves', 0.9815629720687866), 
('1girl', 0.9577146172523499)], 
'character': [
('shimakaze (kantai collection)', 0.9999997615814209), 
('rensouhou-chan', 0.8843141794204712), 
('admiral (kantai collection)', 0.011482828296720982)], 
'copyright': [
('kantai collection', 0.9999978542327881), 
('gundam', 0.0011327610118314624), 
('monster hunter', 0.0009112235275097191)]}
```

