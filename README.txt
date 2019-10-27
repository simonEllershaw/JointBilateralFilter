Joint Bilateral Filter

The 2 python scripts where used to analyse the properties of the bilateral and
joint bilateral filter. BilateralFilter.py uses the OpenCV implementation of the
filter whereas the jointBilateralFilter.py includes my own filter.

The resulting report (Implementation and Analysis of the Bilateral and Joint
Bilateral Filter) is included in this repository.

To run

Install packages using 'pip install -r requirements.txt'
Then call either python file

The input parameters can be changed by editing the variables in the main functions
The jointBilateralFilter.py will output a progress update when it has completed
every 25 lines.
The default output file from jointBilateralFilter.py is
"testImages/jointBilateralFilterOutput.jpg"
bilateralFilter.py will output both a single filtered image and a grid showing
the how the variation of the sigma parameters effects the output.
In the grid sigma_colour changes vertically and sigma_space horizontally
