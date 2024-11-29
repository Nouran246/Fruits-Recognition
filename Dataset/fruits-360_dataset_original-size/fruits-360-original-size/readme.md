# Fruits-360: A dataset of images containing fruits and vegetables #

The archive contains images of fruits, vegetables and nuts at their original (filmed) size. This is the new format of the **Fruits-360 dataset** and is different from the existing dataset whose images have 100x100 pixels in size.

**Filenames in this version do not have the 100x100 suffix. This is how you can make distinction between old files and new files."

## Version: 2024.08.02.0 ##

A dataset of images containing fruits, vegetables and nuts.

## Alternate download ##

The Fruits-360 dataset can be downloaded from: 

**Kaggle** [https://www.kaggle.com/moltean/fruits](https://www.kaggle.com/moltean/fruits)

**GitHub** [https://github.com/fruits-360](https://github.com/fruits-360)

## Dataset properties ##

The total number of images: **TBA**

The training set size: **TBA** images (one fruit or vegetable per image).

The test set size: **TBA** images (one fruit or vegetable per image).

The number of classes: **TBA** (fruits and vegetables).

Image size: variable.

File name format: r?_image_index.jpg (e.g. r0_31.jpg or r1_12.jpg). "r?" stands for rotation axis (first one is r0).

Different fruits (even if they belong to the same varieties) are stored as different objects.

## Repository structure ##

The images have been divided into 3 subsets: [Training](Training), [Validation](Validation) and [Test](Test).
_Training_ contains about 50% of images. _Validation_ and _Test_ each contains about 25% of images.

Folder _Meta_ contains information about the objects in the dataset.
For each object we have additional information like: type, size, ripeness status, water content, fructose content, growth location, etc.

## How to cite ##

Mihai Oltean, [Fruits 360 dataset: new research directions](https://www.researchgate.net/publication/354535752_Fruits_360_dataset_new_research_directions), Technical report, 2021.

## How the dataset was created ##

Fruits and vegetables were planted in the shaft of a low-speed motor (3 rpm) and a short movie of 20 seconds was recorded. 

A _Logitech C920_ camera was used for filming the fruits. This is one of the best webcams available.

Behind the fruits, we placed a white sheet of paper as background. 

However, due to the variations in the lighting conditions, the background was not uniform and we wrote a dedicated algorithm which extract the fruit from the background. This algorithm is of flood fill type: 
we start from each edge of the image and we mark all pixels there, then we mark all pixels found in the neighborhood of the already marked pixels for which the distance between colors is less than a prescribed value. We repeat the previous step until no more pixels can be marked.

All marked pixels are considered as being background (which is then filled with white) and the rest of pixels are considered as belonging to the object.

The maximum value for the distance between 2 neighbor pixels is a parameter of the algorithm and is set (by trial and error) for each movie.

Pictures from the [test-multiple_fruits](test-multiple_fruits) folder were made with a _Nexus 5X_ phone and, more recently, with _iPhone 11_.

## Dataset split ##

The set of images for each object is divided into _Training_, _Validation_ and _Test_ using the following rule:
_k_, _k + 2_ were moved to _Training_,
_k + 1_ was moved to _Validation_,
_k + 3_ was moved to _Test_.

_k_ is the index of the object ( _k_ is multiple of 4).

## Results ##

No numerical experiments are currently performed on this version of the dataset.

## History ##

**This section is not updated. Most info is taken from the 100x100 version.**

Fruits were filmed at the dates given below (YYYY.MM.DD):

2017.02.25 - Apple (golden).

2017.02.28 - Apple (Red Yellow 1, red, golden2), Kiwi, Pear, Grapefruit, Lemon, Orange, Strawberry.

2017.03.05 - Apple (golden3, Braeburn, Granny Smith, red2).

2017.03.07 - Apple (red3).

2017.05.10 - Plum, Peach, Peach flat, Apricot, Nectarine, Pomegranate.

2017.05.27 - Avocado, Papaya, Grape, Cherrie.

2017.12.25 - Carambula, Cactus fruit, Granadilla, Kaki, Kumsquats, Passion fruit, Avocado ripe, Quince.

2017.12.28 - Clementine, Cocos, Mango, Lime, Lychee.

2017.12.31 - Apple Red Delicious, Pear Monster, Grape White.

2018.01.14 - Banana, Grapefruit Pink, Mandarine, Pineapple, Tangelo.

2018.01.19 - Huckleberry, Raspberry.

2018.01.26 - Dates, Maracuja, Plum 2, Salak, Tamarillo.

2018.02.05 - Guava, Grape White 2, Lemon Meyer

2018.02.07 - Banana Red, Pepino, Pitahaya Red.

2018.02.08 - Pear Abate, Pear Williams.

2018.05.22 - Lemon rotated, Pomegranate rotated.

2018.05.24 - Cherry Rainier, Cherry 2, Strawberry Wedge.

2018.05.26 - Cantaloupe (2 varieties).

2018.05.31 - Melon Piel de Sapo.

2018.06.05 - Pineapple Mini, Physalis, Physalis with Husk, Rumbutan.

2018.06.08 - Mulberry, Redcurrant.

2018.06.16 - Cherry Red, Hazelnut, Walnut, Tomato.

2018.06.17 - Cherry Wax (Yellow, Red, Black).

2018.08.19 - Apple Red Yellow 2, Grape Blue, Grape White 2, Grape White 3, Peach 2, Plum 3, Tomato Maroon, Tomato 1-4.

2018.12.20 - Nut Pecan, Pear Kaiser, Tomato Yellow.

2018.12.21 - Banana Lady Finger, Chesnut, Mangostan.

2018.12.22 - Pomelo Sweetie.

2019.04.21 - Apple Crimson Snow, Apple Pink Lady, Blueberry, Kohlrabi, Mango Red, Pear Red, Pepper (Red, Yellow, Green).

2019.06.18 - Beetroot Red, Ginger Root, Nectarine Flat, Nut Forest, Onion Red, Onion Red Peeled, Onion White, Potato Red, Potato Red Washed, Potato Sweet, Potato White.

2019.07.07 - Cauliflower, Eggplant, Pear Forelle, Pepper Orange.

2019.09.22 - Fig, Pear 2, Pear Stone, Watermelon.

## License ##

MIT License

Copyright (c) 2017- [Mihai Oltean](https://mihaioltean.github.io)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
