# NeuralArt
C# implementation of "A Neural Algorithm of Artistic Style".

# How to run
To Perform The Compile.bat. In the Release folder you will see a file NeuralArt.exe. Run it and follow the instructions in the window that opens. In addition, the repository contains a compiled version of the program in the Release folder.

# Speed
On my computer with Intel(P) core(TM) I3 processor 2.40 GHz, iteration in resolution 320x240 takes 90 seconds; in resolution 480 x 360 - 180 seconds; in resolution 640x480 - 320 seconds. Minimum free RAM: 1 GB.

# Results

The number of iterations is 35 for each image.

__Udnie__

![Udnie](https://github.com/PABCSoft/NeuralArt/blob/master/Results/picabia.png)

__Scream__

![Scream](https://github.com/PABCSoft/NeuralArt/blob/master/Results/scream.png)

__Seated Nude__

![Seated Nude](https://github.com/PABCSoft/NeuralArt/blob/master/Results/seated_nude.png)

__Starry night__

![Starry night](https://github.com/PABCSoft/NeuralArt/blob/master/Results/starry_night.png)

__Wreck__

![Wreck](https://github.com/PABCSoft/NeuralArt/blob/master/Results/wreck.png)

# The development of the application:

* The use of L-BFGS-B instead of Adam
* Code optimization using pointers
* Implementation of the language version of the application PascalABC.NET

# Resources & Credits
When developing the application, the library ConvNetCS from Mashmawy was used: https://github.com/mashmawy/ConvNetCS and implementation of the Torch algorithm by Justin Johnson: https://github.com/jcjohnson/neural-style.
