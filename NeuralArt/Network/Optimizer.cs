//*************************************************************************************************
//* Код является разработкой PABCSoft(C. Брыкин Глеб Сергеевич, 2018). Коммерческое использование
//* запрещено. При любом ином использовании указать ссылку на текущий продукт.
//*************************************************************************************************

using System;
using System.IO;
using System.Drawing;
using System.Threading;
using System.Threading.Tasks;
using System.Collections.Generic;

namespace NeuralArt{

  ///<summary>Оптимизатор Adam.</summary>
  public static class Adam{

    ///<summary>Скорость обучения.</summary>
    public static float learning_rate;

    public static float l1_decay;

    public static float l2_decay;

    public static int batch_size;

    public static float momentum;

    public static float ro;

    public static float eps;

    public static float beta1;

    public static float beta2;

    public static int k;

    public static List<float[]> gsum;

    public static List<float[]> xsum;

    ///<summary>Конструктор Adam. Инициализирует все поля.</summary>
    static Adam(){
      learning_rate = 0.5f;
      l1_decay = 0.0f;
      l2_decay = 0.0f;
      batch_size = 1;
      momentum = 0.9f;
      ro = 0.95f;
      eps = 1e-8f;
      beta1 = 0.9f;
      beta2 = 0.999f;
      k = 0;
      gsum = new List<float[]>();
      xsum = new List<float[]>();
    }

    ///<summary>Выполняет обновление параметров по их градиентам.</summary>
    public static void Train(Tensor Value){
      var l2_decay_loss = 0.0f;
      var l1_decay_loss = 0.0f;
      k += 1;
      if(k % batch_size == 0){
        if ((gsum.Count == 0) && (momentum > 0.0f)){
          gsum.Add(new Single[Value.W.Length]);
          xsum.Add(new Single[Value.W.Length]);
        }
      }
      var p = Value.W;
      var g = Value.DW;
      var l2_decay_mul = 0.0f;
      var _l2_decay = l2_decay * l2_decay_mul;
      var _l1_decay = l1_decay * l2_decay_mul;
      var plen = p.Length;
      for(int j = 0; j < plen; j++){
        l2_decay_loss += _l2_decay * p[j] * p[j] / 2.0f;
        l1_decay_loss += _l1_decay * System.Math.Abs(p[j]);
        var l1grad = _l1_decay * ((p[j] > 0.0f) ? 1.0f : -1.0f);
        var l2grad = _l2_decay * (p[j]);
        var gij = (l2grad + l1grad + g[j]) / batch_size;
        var gsumi = gsum[0];
        var xsumi = xsum[0];
        gsumi[j] = gsumi[j] * beta1 + (1 - beta1) * gij;
        xsumi[j] = xsumi[j] * beta2 + (1 - beta2) * gij * gij;
        var biasCorr1 = gsumi[j] * (1 - System.Math.Pow(beta1, k));
        var biasCorr2 = xsumi[j] * (1 - System.Math.Pow(beta2, k));
        var dx = -learning_rate * biasCorr1 / (System.Math.Sqrt(biasCorr2) + eps);
        if (System.Math.Abs(dx) < 20.0f){
          p[j] += (float)dx;
        }
        else{
          p[j] += (dx > 0) ? (20.0f) : (-20.0f);
        }
      }
    }

  }

}