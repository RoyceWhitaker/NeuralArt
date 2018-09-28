//*************************************************************************************************
//* Код является разработкой PABCSoft(C. Брыкин Глеб Сергеевич, 2018). Коммерческое использование
//* запрещено. При любом ином использовании указать ссылку на текущий продукт.
//*************************************************************************************************

using System;
using System.IO;
using System.Drawing;
using System.Threading;
using System.Threading.Tasks;

namespace NeuralArt{

  ///<summary>Интерфейс слоя нейросети.</summary>
  public interface Layer{

    ///<summary>Входные данные слоя.</summary>
    Tensor Input{get; set;}

    ///<summary>Выходные данные.</summary>
    Tensor Output{get; set;}

    ///<summary>Ширина входного тензора.</summary>
    int InputWidth{get; set;}

    ///<summary>Высота входного тензора.</summary>
    int InputHeight{get; set;}

    ///<summary>Глубина входного тензора.</summary>
    int InputDepth{get; set;}

    ///<summary>Ширина выходного тензора.</summary>
    int OutputWidth{get; set;}

    ///<summary>Высота выходного тензора.</summary>
    int OutputHeight{get; set;}

    ///<summary>Глубина выходного тензора.</summary>
    int OutputDepth{get; set;}

    ///<summary>Метод прямого распространения через слой.</summary>
    ///<param name="input">Входные данные.</param>
    Tensor Forward(Tensor input);

    ///<summary>Метод обратного распространения (градиента) через слой.</summary>
    void Backward();

  }

  // В архитектуре используемой нейросети (VGG-16) все свёртки имеют фильтры размером 3x3. Для оптимизации,
  // эти значения интегрированы в код.

  ///<summary>Свёрточный слой.</summary>
  public sealed class ConvLayer:Layer{

    ///<summary>Ширина входного тензора.</summary>
    public int InputWidth{get; set;}

    ///<summary>Высота входного тензора.</summary>
    public int InputHeight{get; set;}

    ///<summary>Глубина входного тензора.</summary>
    public int InputDepth{get; set;}

    ///<summary>Ширина выходного тензора.</summary>
    public int OutputWidth{get; set;}

    ///<summary>Высота выходного тензора.</summary>
    public int OutputHeight{get; set;}

    ///<summary>Глубина выходного тензора.</summary>
    public int OutputDepth{get; set;}

    ///<summary>Входные данные.</summary>
    public Tensor Input{get; set;}

    ///<summary>Выходные данные.</summary>
    public Tensor Output{get; set;}

    ///<summary>Фильтры.</summary>
    public Tensor[] Filters;

    ///<summary>Смещения.</summary>
    public Tensor Biases;

    ///<summary>Инициализирует свёрточный слой с заданными фильтрами и смещениями.</summary>
    ///<param name="filters">Фильтры.</param>
    ///<param name="biases">Смещения.</param>
    public ConvLayer(Tensor[] filters, Tensor biases){
      this.InputWidth = 0;
      this.InputHeight = 0;
      this.InputDepth = 0;
      this.OutputWidth = 0;
      this.OutputHeight = 0;
      this.OutputDepth = filters.Length;
      this.Filters = filters;
      this.Biases = biases;
    }

    ///<summary>Прямое распространение через свёрточный слой.</summary>
    ///<param name="input">Входные данные.</param>
    public Tensor Forward(Tensor input){
      this.InputWidth = input.Width;
      this.InputHeight = input.Height;
      this.InputDepth = input.Depth;
      this.OutputWidth = input.Width;
      this.OutputHeight = input.Height;
      this.Input = input;
      var Result = new Tensor(this.OutputWidth, this.OutputHeight, this.OutputDepth, true);
      Parallel.For(0, this.OutputDepth, (int d) => {
        for (int ay = 0; ay < this.OutputHeight; ay++){
          var y = ay - 1;
          var f = this.Filters[d];
          for (var ax = 0; ax < this.OutputWidth; ax++){
            var x = ax - 1;
            var a = 0.0;
            for (byte fy = 0; fy < 3; fy++){
              var oy = y + fy;
              for (var fx = 0; fx < 3; fx++){
                var ox = x + fx;
                if ((oy >= 0) && (oy < this.InputHeight) && (ox >= 0) && (ox < this.InputWidth)){
                  for (var fd = 0; fd < f.Depth; fd++){
                    a += f.W[((3 * fy) + fx) * f.Depth + fd] * input.W[((this.InputWidth * oy) + ox) * input.Depth + fd];
                  }
                }
              }
            }
            a += this.Biases.W[d];
            Result.SetW(ax, ay, d, (float)a);
          }
        }
      });
      this.Output = Result;
      return Result;
    }

    ///<summary>Обратное распространение (градиента) через свёрточный слой.</summary>
    public void Backward(){
      var V = this.Input;
      V.DW = new float[V.W.Length];
      var inputWidth = V.Width;
      var inputHeight = V.Height;
      System.Threading.Tasks.Parallel.For(0, this.OutputDepth, (int d) => {
        var f = this.Filters[d];
        var x = -1;
        var y = -1;
        for (var ay = 0; ay < this.OutputHeight; y += 1, ay++){
          x = -1;
          for (var ax = 0; ax < this.OutputWidth; x += 1, ax++){
            var chain_grad = this.Output.GetDW(ax, ay, d);
            for (var fy = 0; fy < 3; fy++){
              var oy = y + fy;
              for (byte fx = 0; fx < 3; fx++){
                var ox = x + fx;
                if ((oy >= 0) && (oy < inputHeight) && (ox >= 0) && (ox < inputWidth)){
                  for (var fd = 0; fd < f.Depth; fd++){
                    var ix1 = ((inputWidth * oy) + ox) * V.Depth + fd;
                    var ix2 = ((3 * fy) + fx) * f.Depth + fd;
                    V.DW[ix1] += f.W[ix2] * chain_grad;
                  }
                }
              }
            }
          }
        }
      });
    }

  }

  ///<summary>Слой линейной ректификации (ReLU).</summary>
  public sealed class ReluLayer:Layer{

    ///<summary>Ширина входного тензора.</summary>
    public int InputWidth{get; set;}

    ///<summary>Высота входного тензора.</summary>
    public int InputHeight{get; set;}

    ///<summary>Глубина входного тензора.</summary>
    public int InputDepth{get; set;}

    ///<summary>Ширина выходного тензора.</summary>
    public int OutputWidth{get; set;}

    ///<summary>Высота выходного тензора.</summary>
    public int OutputHeight{get; set;}

    ///<summary>Глубина выходного тензора.</summary>
    public int OutputDepth{get; set;}

    ///<summary>Входные данные.</summary>
    public Tensor Input{get; set;}

    ///<summary>Выходные данные.</summary>
    public Tensor Output{get; set;}

    ///<summary>Инициализирует слой линейной ректификации (ReLU).</summary>
    public ReluLayer(){
      this.InputWidth = 0;
      this.InputHeight = 0;
      this.InputDepth = 0;
      this.OutputWidth = 0;
      this.OutputHeight = 0;
      this.OutputDepth = 0;
    }

    ///<summary>Прямое распространение через слой линейной ректификации (ReLU).</summary>
    ///<param name="input">Входные данные.</param>
    public Tensor Forward(Tensor input){
      this.Input = input;
      this.InputWidth = input.Width;
      this.InputHeight = input.Height;
      this.InputDepth = input.Depth;
      this.OutputWidth = input.Width;
      this.OutputHeight = input.Height;
      this.OutputDepth = input.Depth;
      var Result = new Tensor(input.Width, input.Height, input.Depth, true);
      Parallel.For(0, OutputDepth, (int d) => {
        for (int y = 0; y < OutputHeight; y++){
          for (int x = 0; x < OutputWidth; x++){
            var v = input.GetW(x, y, d);
            Result.SetW(x, y, d, ((v > 0.0f) ? (v) : (0.0f)));
          }
        }
      });
      this.Output = Result;
      return Result;
    }

    ///<summary>Обратное распространение (градиента) через слой линейной ректификации (ReLU).</summary>
    public void Backward(){
      var V = this.Input;
      var V2 = this.Output;
      V.DW = new float[V.W.Length];
      Parallel.For(0, this.OutputDepth, (int d) => {
        for (int y = 0; y < this.OutputHeight; y++){
          for (int x = 0; x < this.OutputWidth; x++){
            if (V2.GetW(x, y, d) <= 0.0f){
              V.SetDW(x, y, d, 0.0f);
            }
            else{
              V.SetDW(x, y, d, V2.GetDW(x, y, d));
            }
          }
        }
      });
    }

  }

  // В архитектуре используемой нейросети (VGG-16) все пулинги имеют окна размером 2x2. Для оптимизации,
  // эти значения интегрированы в код.

  ///<summary>Слой устредняющего пулинга/подвыборки (Averaging Pooling).</summary>
  public sealed class AvgPoolLayer:Layer{

    ///<summary>Ширина входного тензора.</summary>
    public int InputWidth{get; set;}

    ///<summary>Высота входного тензора.</summary>
    public int InputHeight{get; set;}

    ///<summary>Глубина входного тензора.</summary>
    public int InputDepth{get; set;}

    ///<summary>Ширина выходного тензора.</summary>
    public int OutputWidth{get; set;}

    ///<summary>Высота выходного тензора.</summary>
    public int OutputHeight{get; set;}

    ///<summary>Глубина выходного тензора.</summary>
    public int OutputDepth{get; set;}

    ///<summary>Входные данные.</summary>
    public Tensor Input{get; set;}

    ///<summary>Выходные данные.</summary>
    public Tensor Output{get; set;}

    ///<summary>Инициализирует слой устредняющего пулинга/подвыборки (Averaging Pooling).</summary>
    public AvgPoolLayer(){
      this.InputWidth = 0;
      this.InputHeight = 0;
      this.InputDepth = 0;
      this.OutputWidth = 0;
      this.OutputHeight = 0;
      this.OutputDepth = 0;
    }

    ///<summary>Прямое распространение через слой усредняющего пулинга/подвыборки (Averaging Pooling).</summary>
    ///<param name="input">Входные данные.</param>
    public Tensor Forward(Tensor input){
      this.Input = input;
      this.InputWidth = input.Width;
      this.InputHeight = input.Height;
      this.InputDepth = input.Depth;
      this.OutputWidth = input.Width / 2;
      this.OutputHeight = input.Height / 2;
      this.OutputDepth = input.Depth;
      var A = new Tensor(this.OutputWidth, this.OutputHeight, this.OutputDepth, true);
      Parallel.For(0, this.OutputDepth, (int d) => {
        for (var ax = 0; ax < this.OutputWidth; ax++){
          var x = 2 * ax;
          for (var ay = 0; ay < this.OutputHeight;  ay++){
            var y = 2 * ay; 
            float a = 0.0f;
            for (byte fx = 0; fx < 2; fx++){
              for (byte fy = 0; fy < 2; fy++){
                var oy = y + fy;
                var ox = x + fx;
                if ((oy >= 0) && (oy < input.Height) && (ox >= 0) && (ox < input.Width)){
                  var v = input.GetW(ox, oy, d);
                  a += v;
                }
              }
            }
            A.SetW(ax, ay, d, a / 4);
          }
        }
      });
      this.Output = A;
      return A;
    }

    ///<summary>Обратное распространение (градиента) через слой устредняющего пулинга/подвыборки (Averaging Pooling).</summary>
    public void Backward(){
      var V = this.Input;
      V.DW = new float[V.W.Length];
      var A = this.Output;
      Parallel.For(0, this.OutputDepth, (int d) => {
        for (var ax = 0; ax < this.OutputWidth; ax++){
          var x = 2 * ax;
          for (var ay = 0; ay < this.OutputHeight; ay++){
            var y = 2 * ay; 
            float a = A.GetDW(ax, ay, d) / 4.0f;
            for (byte fx = 0; fx < 2; fx++){
              for (byte fy = 0; fy < 2; fy++){
                var oy = y + fy;
                var ox = x + fx;
                if ((oy >= 0) && (oy < V.Height) && (ox >= 0) && (ox < V.Width)){
                  V.AddDW(ox, oy, d, a);
                }
              }
            }
          }
        }
      });
    }

  }

  ///<summary>Loss - network. Нейросеть VGG - 16.</summary>
  public sealed class VGG16{

    ///<summary>Слои нейросети.</summary>
    private Layer[] Layers;

    ///<summary>Матрица Грамма для стилевой карты признаков из первого блока нейросети.</summary>
    private Tensor Style1;

    ///<summary>Матрица Грамма для стилевой карты признаков из второго блока нейросети.</summary>
    private Tensor Style2;

    ///<summary>Матрица Грамма для стилевой карты признаков из третьего блока нейросети.</summary>
    private Tensor Style3;

    ///<summary>Матрица Грамма для стилевой карты признаков из четвёртого блока нейросети.</summary>
    private Tensor Style4;

    ///<summary>Карта признаков контентного изображения.</summary>
    private Tensor Content;

    ///<summary>Читает фильтры свёрточного слоя из потока через BinaryReader.</summary>
    ///<param name="n">Количество фильтров.</param>
    ///<param name="d">Глубина фильтров.</param>
    ///<param name="br">Оболочка для потока, из которого читаются данные.</param>
    private static Tensor[] LoadFilters(int n, int d, BinaryReader br){
      var Result = new Tensor[n];
      for(int i = 0; i < n; i++){
        Result[i] = new Tensor(3, 3, d, false);
        var f = Result[i];
        for(int z = 0; z < d; z++){
          for(byte y = 0; y < 3; y++){
            for(byte x = 0; x < 3; x++){
              f.SetW(x, y, z, br.ReadSingle());
            }
          }
        }
      }
      return Result;
    }

    ///<summary>Читает смещения свёрточного слоя из потока через BinaryReader.</summary>
    ///<param name="n">Количество значений.</param>
    ///<param name="br">Оболочка для потока, из которого читаются данные.</param>
    private static Tensor LoadBiases(int n, BinaryReader br){
      var Result = new Tensor(1, 1, n, false);
      var f = Result.W;
      for(int i = 0; i < n; i++){
        f[i] = br.ReadSingle();
      }
      return Result;
    }

    ///<summary>Делегат для события завершения итерации.</summary>
    public delegate void IterationHandler(int iter, float loss);

    ///<summary>Событие завершения итерации.</summary>
    public event IterationHandler OnIterationDone;

    ///<summary>Инициализирует Loss - нейросеть VGG16, считывая данные свёрточных слоёв из потока s.</summary>
    ///<param name="s">Поток, из которого будут считаны данные свёрточных слоёв (фильтры и смещения).</param>
    public VGG16(Stream s){
      var br = new BinaryReader(s);
      this.Layers = new Layer[19];
      // block 1
      this.Layers[0] = new ConvLayer(LoadFilters(64, 3, br), LoadBiases(64, br));
      this.Layers[1] = new ReluLayer();
      this.Layers[2] = new ConvLayer(LoadFilters(64, 64, br), LoadBiases(64, br));
      this.Layers[3] = new ReluLayer();
      this.Layers[4] = new AvgPoolLayer();
      // block 2
      this.Layers[5] = new ConvLayer(LoadFilters(128, 64, br), LoadBiases(128, br));
      this.Layers[6] = new ReluLayer();
      this.Layers[7] = new ConvLayer(LoadFilters(128, 128, br), LoadBiases(128, br));
      this.Layers[8] = new ReluLayer();
      this.Layers[9] = new AvgPoolLayer();
      // block 3
      this.Layers[10] = new ConvLayer(LoadFilters(256, 128, br), LoadBiases(256, br));
      this.Layers[11] = new ReluLayer();
      this.Layers[12] = new ConvLayer(LoadFilters(256, 256, br), LoadBiases(256, br));
      this.Layers[13] = new ReluLayer();
      this.Layers[14] = new ConvLayer(LoadFilters(256, 256, br), LoadBiases(256, br));
      this.Layers[15] = new ReluLayer();
      this.Layers[16] = new AvgPoolLayer();
      // block 4
      this.Layers[17] = new ConvLayer(LoadFilters(512, 256, br), LoadBiases(512, br));
      this.Layers[18] = new ReluLayer();
    }

    ///<summary>Прямое распространение через нейросеть.</summary>
    ///<param name="input">Входные данные.</param>
    public void Forward(Tensor input){
      var act = this.Layers[0].Forward(input);
      for(byte i = 1; i < 19; i++){
        act = this.Layers[i].Forward(act);
      }
    }

    // Стилевые слои: ReLU1_1, ReLU2_1, ReLU3_1, ReLU4_1.

    ///<summary>Фиксирует матрицы Грамма для стилевых признаков.</summary>
    ///<param name="S">Стилевое изображение.</param>
    public void FixStyle(Tensor S){
      this.Forward(S);
      this.Style1 = Math.Gram_Matrix(this.Layers[1].Output);
      this.Style2 = Math.Gram_Matrix(this.Layers[6].Output);
      this.Style3 = Math.Gram_Matrix(this.Layers[11].Output);
      this.Style4 = Math.Gram_Matrix(this.Layers[18].Output);
    }

    // Контентный слой: ReLU3_3.

    ///<summary>Фиксирует контентные признаки.</summary>
    ///<param name="C">Контентное изображение.</param>
    public void FixContent(Tensor C){
      this.Forward(C);
      this.Content = this.Layers[15].Output.Clone();
    }

    ///<summary>Обратное распространение ошибки(градиента) и её попутное вычисление.</summary>
    public float Backward(){
      this.Layers[18].Output.DW = new float[this.Layers[18].Output.W.Length];
      var Result = Losses.StyleLoss(this.Layers[18].Output, this.Style4, 0.2e6f);
      this.Layers[18].Backward();
      this.Layers[17].Backward();
      this.Layers[16].Backward();
      Result += Losses.ContentLoss(this.Layers[15].Output, this.Content, 8.0f);
      this.Layers[15].Backward();
      this.Layers[14].Backward();
      this.Layers[13].Backward();
      this.Layers[12].Backward();
      Result += Losses.StyleLoss(this.Layers[11].Output, this.Style3, 0.2e6f);
      this.Layers[11].Backward();
      this.Layers[10].Backward();
      this.Layers[9].Backward();
      this.Layers[8].Backward();
      this.Layers[7].Backward();
      Result += Losses.StyleLoss(this.Layers[6].Output, this.Style2, 0.2e6f);
      this.Layers[6].Backward();
      this.Layers[5].Backward();
      this.Layers[4].Backward();
      this.Layers[3].Backward();
      this.Layers[2].Backward();
      Result += Losses.StyleLoss(this.Layers[1].Output, this.Style1, 0.2e6f);
      this.Layers[1].Backward();
      this.Layers[0].Backward();
      return Result;
    }

    ///<summary>Запускает итеративный процесс. Количество итераций: 1000.</summary>
    ///<param name="X">Начальное изображение (холст).</param>
    public void StartIterativeProcess(Tensor X){
      for(int i = 0; i < 1000; i++){
        this.Forward(X);
        var l = this.Backward();
        l += Losses.TVFLoss(X, 1f);
        Adam.Train(X);
        if(OnIterationDone != null){
          OnIterationDone(i, l);
        }
      }
    }

  }

}
