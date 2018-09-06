# NeuralArt
C# implementation of "A Neural Algorithm of Artistic Style".

# Как запустить
Выполнить Compile.bat. В папке Release появится файл NeuralArt.exe. Запустить его и следовать инструкциям в открывшемся окне.

# Время обработки
На моём компьютере с Intel(R) Core(TM) i3 2.40 GHz, итерация в разрешении 320x240 занимает 90 сек; в разрешении 480x360 - 180 сек; в разрешении 640x480 - 320 сек.

# Результаты
Udnie

![Udnie](https://github.com/PABCSoft/NeuralArt/blob/master/Results/picabia.png)

Scream

![Scream](https://github.com/PABCSoft/NeuralArt/blob/master/Results/scream.png)

Seated Nude

![Seated Nude](https://github.com/PABCSoft/NeuralArt/blob/master/Results/seated_nude.png)

Starry night

![Starry night](https://github.com/PABCSoft/NeuralArt/blob/master/Results/starry_night.png)

Wreck

![Wreck](https://github.com/PABCSoft/NeuralArt/blob/master/Results/wreck.png)

# Ресурсы & Благодарности
При разработке приложения использовалась библиотека ConvNetCS от Mashmawy: https://github.com/mashmawy/ConvNetCS и реализация алгоритма на Torch от Justin Johnson: https://github.com/jcjohnson/neural-style.
