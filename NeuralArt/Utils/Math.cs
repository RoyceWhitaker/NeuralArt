//*************************************************************************************************
//* Код является разработкой PABCSoft(C. Брыкин Глеб Сергеевич, 2018). Коммерческое использование
//* запрещено. При любом ином использовании указать ссылку на текущий продукт.
//*************************************************************************************************

using System;
using System.Drawing;
using System.Threading;
using System.Threading.Tasks;
using System.Drawing.Imaging;

namespace NeuralArt{

  ///<summary>Предоставляет математические операции над матрицами.</summary>
  public static class Math{

    ///<summary>Разворачивает трёхмерный массив в двумерный.</summary>
    ///<param name="T">Трёхмерный входной тензор.</param>
    public static Tensor Flat(Tensor T){
      var PNum = Environment.ProcessorCount;
      var Result = new Tensor(T.Width * T.Height, T.Depth, 1, true);
      var TaskPart = T.Depth / PNum;
      var width = T.Width;
      var height = T.Height;
      Parallel.For(0, PNum, (int p) => {
        for(int d = p * TaskPart; d < (p + 1) * TaskPart; d++){
          int i = 0;
          for(int y = 0; y < height; y++){
            for(int x = 0; x < width; x++){
              Result.SetW(i, d, 0, T.GetW(x, y, d));
              i += 1;
            }
          }
        }
      });
      for(int d = PNum * TaskPart; d < T.Depth; d++){
        int i = 0;
        for(int y = 0; y < height; y++){
          for(int x = 0; x < width; x++){
            Result.SetW(i, d, 0, T.GetW(x, y, d));
            i += 1;
          }
        }
      }
      return Result;
    }

    ///<summary>Транспонирует тензор в его двумерном представлении.</summary>
    ///<param name="T">Входной тензор в двумерном представлении.</param>
    public static Tensor Transpose2D(Tensor T){
      var PNum = Environment.ProcessorCount;
      var Result = new Tensor(T.Height, T.Width, 1, true);
      var TaskPart = T.Height / PNum;
      var width = T.Width;
      Parallel.For(0, PNum, (int p) => {
        for(int y = TaskPart * p; y < TaskPart * (p + 1); y++){
          for(int x = 0; x < width; x++){
            Result.SetW(y, x, 0, T.GetW(x, y, 0));
          }
        }
      });
      for(int y = PNum * TaskPart; y < T.Height; y++){
        for(int x = 0; x < width; x++){
          Result.SetW(y, x, 0, T.GetW(x, y, 0));
        }
      }
      return Result;
    }

    ///<summary>Выполняет умножение двух тензоров в двумерном представлении.</summary>
    ///<param name="A">Первая матрица - множитель.</param>
    ///<param name="B">Вторая матрица - множитель.</param>
    public static Tensor MatMul2D(Tensor A, Tensor B){
      var PNum = Environment.ProcessorCount;
      var Result = new Tensor(A.Width, B.Height, 1, true);
      var TaskPart = A.Width / PNum;
      Parallel.For(0, PNum, (int p) => {
        for (int i = TaskPart * p; i < TaskPart * (p + 1); i++){
          for (int j = 0; j < B.Height; j++){
            for (int k = 0; k < B.Width; k++){
              Result.AddW(i, j, 0, A.GetW(i, k, 0) * B.GetW(k, j, 0));
            }
          }
        }
      });
      for (int i = TaskPart * PNum; i < A.Width; i++){
        for (int j = 0; j < B.Height; j++){
          for (int k = 0; k < B.Width; k++){
            Result.AddW(i, j, 0, A.GetW(i, k, 0) * B.GetW(k, j, 0));
          }
        }
      }
      return Result;
    }

    ///<summary>Очищает изображение от шума, сглаживает.</summary>
    ///<param name="Inp">Входное изображение.</param>
    public static Tensor Denoise(Tensor Inp){
      var Result = new Tensor(Inp.Width, Inp.Height, 3, true);
      for(int y = 1; y < Inp.Height - 1; y++){
        for(int x = 1; x < Inp.Width - 1; x++){
          var SumR = 0.0f;
          var SumG = 0.0f;
          var SumB = 0.0f;
          for(sbyte i = -1; i < 2; i++){
            for(sbyte j = -1; j < 2; j++){
              SumR += Inp.GetW(x + j, y + i, 0);
              SumG += Inp.GetW(x + j, y + i, 1);
              SumB += Inp.GetW(x + j, y + i, 2);
            }
          }
          Result.SetW(x - 1, y - 1, 0, SumR / 9.0f);
          Result.SetW(x - 1, y - 1, 1, SumG / 9.0f);
          Result.SetW(x - 1, y - 1, 2, SumB / 9.0f);
        }
      }
      return Result;
    }

    ///<summary>Вычисляет матрицу Грамма (Граммиан) для тензора.</summary>
    ///<param name="F">Входной тензор.</param>
    public static Tensor Gram_Matrix(Tensor F){
      var Ft = Flat(F);
      return MatMul2D(Transpose2D(Ft), Ft);
    }

    ///<summary>Выполняет нормализацию изображения.</summary>
    ///<param name="BMP">Изображение.</param>
    public static unsafe System.Drawing.Bitmap RestoreColors(System.Drawing.Bitmap BMP){
      var Result = new System.Drawing.Bitmap(BMP.Width, BMP.Height);
      var arrRed = new System.UInt16[256];
      var arrGreen = new System.UInt16[256];
      var arrBlue = new System.UInt16[256];
      var Width = BMP.Width;
      var Height = BMP.Height;
      var BD = BMP.LockBits(new System.Drawing.Rectangle(0, 0, Width, Height), System.Drawing.Imaging.ImageLockMode.ReadOnly, System.Drawing.Imaging.PixelFormat.Format24bppRgb);
      System.Threading.Tasks.Parallel.For(0, Height, (System.Int32 y)=>{
        var Address = (System.Byte*)(BD.Scan0.ToInt32() + y * BD.Stride);
        for(System.Int32 x = 0; x < Width; x++){
          lock (arrBlue){
            arrBlue[*Address] += 1;
          }
          Address += 1;
          lock (arrGreen){
            arrGreen[*Address] += 1;
          }
          Address += 1;
          lock (arrRed){
            arrRed[*Address] += 1;
          }
          Address += 1;
        }
      });
      System.Double q = Width * Height * 0.01;
      System.UInt16 newMinR = 0;
      System.UInt16 newMaxR = 0;
      System.UInt16 newMinG = 0;
      System.UInt16 newMaxG = 0;
      System.UInt16 newMinB = 0;
      System.UInt16 newMaxB = 0;
      System.UInt64 s = 0;
      for(System.Byte i = 0; i <= 255; i++){
        s += arrRed[i];
        if (s >= q){
          newMinR = i;
          break;
        }
      }
      s = 0;
      for(System.Byte i = 255; i >= 0; i--){
        s += arrRed[i];
        if (s >= q){
          newMaxR = i;
          break;
        }
      }
      s = 0;
      for(System.Byte i = 0; i <= 255; i++){
        s += arrGreen[i];
        if (s >= q){
          newMinG = i;
          break;
        }
      }
      s = 0;
      for(System.Byte i = 255; i >= 0; i--){
        s += arrGreen[i];
        if (s >= q){
          newMaxG = i;
          break;
        }
      }
      s = 0;
      for(System.Byte i = 0; i <= 255; i++){
        s += arrBlue[i];
        if (s >= q){
          newMinB = i;
          break;
        }
      }
      s = 0;
      for(System.Byte i = 255; i >= 0; i--){
        s += arrBlue[i];
        if (s >= q){
          newMaxB = i;
          break;
        }
      }
      var BW_BD = Result.LockBits(new System.Drawing.Rectangle(0, 0, Width, Height), System.Drawing.Imaging.ImageLockMode.WriteOnly, System.Drawing.Imaging.PixelFormat.Format24bppRgb);
      System.Threading.Tasks.Parallel.For(0, Height, (System.Int32 y)=>{
        System.Byte* BW_Address;
        System.Byte* In_Address;
        System.Int16 newR = 0;
        System.Int16 newG = 0;
        System.Int16 newB = 0;
        In_Address = (System.Byte*)(BD.Scan0.ToInt32() + y * BD.Stride);
        BW_Address = (System.Byte*)(BW_BD.Scan0.ToInt32() + y * BW_BD.Stride);
        for(System.Int32 x = 0; x < Width; x++){
          newB = System.Convert.ToInt16(System.Math.Round((System.Single)(*In_Address - newMinB) * 255.0f / (newMaxB - newMinB+1)));
          if (newB < 0){
            newB = 0;
          }
          if (newB > 255){
            newB = 255;
          }
          In_Address+=1;
          newG = System.Convert.ToInt16(System.Math.Round((System.Single)(*In_Address - newMinG) * 255.0f / (newMaxG - newMinG+1)));
          if(newG < 0){
            newG = 0;
          }
          if(newG > 255){
            newG = 255;
          }
          In_Address+=1;
          newR = System.Convert.ToInt16(System.Math.Round((System.Single)(*In_Address - newMinR) * 255.0f / (newMaxR - newMinR+1)));
          if (newR < 0){
            newR = 0;
          }
          if (newR > 255){
            newR = 255;
          }
          In_Address+=1;
          *BW_Address = (System.Byte)newB;
          BW_Address += 1;
          *BW_Address = (System.Byte)newG;
          BW_Address += 1;
          *BW_Address = (System.Byte)newR;
          BW_Address += 1;
        }
      });
      BMP.UnlockBits(BD);
      Result.UnlockBits(BW_BD);
      arrRed = null;
      arrGreen = null;
      arrBlue = null;
      System.GC.Collect();
      return Result;
    }

  }

  ///<summary>Предоставляет методы для преобразований между System.Drawing.Bitmap и классом Tensor.</summary>
  public static class IOConverters{

    ///<summary>Преобразует изображение в тензор.</summary>
    ///<param name="Image">Изображение.</param>
    public static unsafe Tensor ImageToTensor(Bitmap Image){
      var Result = new Tensor(Image.Width, Image.Height, 3, true);
      var BD = Image.LockBits(new Rectangle(0, 0, Image.Width, Image.Height), ImageLockMode.ReadOnly, PixelFormat.Format24bppRgb);
      var w = Image.Width;
      Parallel.For(0, Image.Height, (int y) => {
        var Addr = (byte*)(BD.Scan0.ToInt32() + BD.Stride * y);
        for(int x = 0; x < w; x++){
          Result.SetW(x, y, 0, (float)(*Addr));
          Addr+=1;
          Result.SetW(x, y, 1, (float)(*Addr));
          Addr+=1;
          Result.SetW(x, y, 2, (float)(*Addr));
          Addr+=1;
        }
      });
      Image.UnlockBits(BD);
      return Result;
    }

    ///<summary>Преобразует тензор в изображение.</summary>
    ///<param name="Image">Тензор.</param>
    public static unsafe Bitmap TensorToImage(Tensor img){
      var result = new Bitmap(img.Width, img.Height);
      var min = 99999999.0f;
      var max = 0.0f;
      for(byte d = 0; d < 3; d++){
        for(int x = 0; x < img.Width; x++){
          for(int y = 0; y < img.Height; y++){
            var _v = img.GetW(x, y, d);
            if (_v < min){
              min = _v;
            }
            if (_v > max){
              max = _v;
            }
          }
        }
      }
      var BD = result.LockBits(new Rectangle(0, 0, result.Width, result.Height), ImageLockMode.WriteOnly, PixelFormat.Format24bppRgb);
      for(int y = 0; y < img.Height; y++){
        byte* Addr = (byte*)(BD.Scan0.ToInt32() + BD.Stride * y);
        for(int x = 0; x < img.Width; x++){
          var r = ((img.GetW(x, y, 2) - min) / (max - min)) * 255.0f;
          if (r < 0.0f){
            r = 0.0f;
          }
          if (r > 255.0f){
            r = 255.0f;
          }
          var g = ((img.GetW(x, y, 1) - min) / (max - min)) * 255.0f;
          if (g < 0.0f){
            g = 0.0f;
          }
          if (g > 255.0f){
            g = 255.0f;
          }
          var b = ((img.GetW(x, y, 0) - min) / (max - min)) * 255.0f;
          if (b < 0.0f){
            b = 0.0f;
          }
          if (b > 255.0f){
            b = 255.0f;
          }
          *Addr = (byte)b;
          Addr += 1;
          *Addr = (byte)g;
          Addr += 1;
          *Addr = (byte)r;
          Addr += 1;
        }
      }
      result.UnlockBits(BD);
      BD = null;
      return result;
    }

  }

}