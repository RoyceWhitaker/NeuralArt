# NeuralArt
C# implementation of "A Neural Algorithm of Artistic Style".

# Как запустить
Выполнить Compile.bat. В папке Release появится файл NeuralArt.exe. Запустить его и следовать инструкциям в открывшемся окне. Кроме того, репозиторий содержит откомпилированную версию программы в папке Release.

# Время обработки
На моём компьютере с Intel(R) Core(TM) i3 2.40 GHz, итерация в разрешении 320x240 занимает 90 сек; в разрешении 480x360 - 180 сек; в разрешении 640x480 - 320 сек. Минимальный объём свободной оперативной памяти: 1 ГБ.

# Результаты

Количество итераций - 35 для каждого изображения.

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

# Развитие приложения:

* Использование L-BFGS-B вместо Adam
* Оптимизация кода с использованием указателей
* Реализация версии приложения на языке PascalABC.NET

# Ресурсы & Благодарности
При разработке приложения использовалась библиотека ConvNetCS от Mashmawy: https://github.com/mashmawy/ConvNetCS и реализация алгоритма на Torch от Justin Johnson: https://github.com/jcjohnson/neural-style. Очень полезной оказалась реализация от Kautenja: https://github.com/Kautenja/neural-style-transfer/
