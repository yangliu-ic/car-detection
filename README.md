# U-Net Car Detection

Implementation of U-Net in Tensorflow for car detection.

Original Paper
- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)

## Summary

<table>
    <tr>
        <th>Input/Output</th>
        <th>Shape</th>
        <th>Explanation</th>
    </tr>
    <tr>
        <td>X: 3-D Tensor</td>
        <td>(512, 512, 3)</td>
        <td>RGB image in an array</td>
    </tr>
    <tr>
        <td>y: 3-D Tensor</td>
        <td>(512, 512, 1)</td>
        <td>Binarized image: Bacground is 0;<br />Vehicle is masked as 255</td>
    </tr>
</table>


## Loss Function

Maximize IOU

```
    (intersection of prediction & grount truth)
    -------------------------------------------
        (union of prediction & ground truth)
```
