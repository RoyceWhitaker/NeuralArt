//*************************************************************************************************
//* Код является разработкой PABCSoft(C. Брыкин Глеб Сергеевич, 2018). Коммерческое использование
//* запрещено. При любом ином использовании указать ссылку на текущий продукт.
//*************************************************************************************************

using System;
using System.Drawing;
using System.Reflection;
using System.Windows.Forms;

namespace NeuralArt{

  ///<summary>Главная форма приложения.</summary>
  public sealed partial class MainForm:Form{

    ///<summary>Поле настроек контента.</summary>
    public GroupBox ContentSettings;

    ///<summary>Поле настроек результата.</summary>
    public GroupBox ResultSettings;

    ///<summary>Поле настроек стиля.</summary>
    public GroupBox StyleSettings;

    ///<summary>Контентное изображение.</summary>
    public PictureBox ContentImageBox;

    ///<summary>Стилизованное изображение.</summary>
    public PictureBox ResultImageBox;

    ///<summary>Стилевое изображение.</summary>
    public PictureBox StyleImageBox;

    ///<summary>Кнопка открытия контентного изображения.</summary>
    public Button OpenContent;

    ///<summary>Кнопка сохранения стилизованного изображения.</summary>
    public Button SaveResult;

    ///<summary>Кнопка открытия стилевого изображения.</summary>
    public Button OpenStyle;

    ///<summary>Размеры контентного изображения.</summary>
    public ComboBox ContentSizes;

    ///<summary>Размеры стилевого изображения.</summary>
    public ComboBox StyleSizes;

    ///<summary>Флажок, указывающий, требуется ли нормализация результатирующего изображения.</summary>
    public CheckBox ResultNorm;

    ///<summary>Кнопка запуска процесса.</summary>
    public Button StartProcess;

    ///<summary>Вывод параметров итерации.</summary>
    public Label Iteration;

    ///<summary>Кнопка остановки процесса.</summary>
    public Button StopProcess;

    ///<summary>Возвращает размер для выбранного параметра Content/StyleSizes.</summary>
    ///<param name="Index">Индекс выбранного параметра.</param>
    public static Size GetSize(int Index){
      switch(Index){
        case 0:{
          return new Size(320, 240);
        }
        case 1:{
          return new Size(480, 320);
        }
        case 2:{
          return new Size(640, 480);
        }
        case 3:{
          return new Size(800, 600);
        }
        case 4:{
          return new Size(1280, 720);
        }
      }
      return new Size(640, 480);
    }

    ///<summary>Инициализирует форму и элементы управления формы.</summary>
    public void Initialize(){
      //-> MainForm
      this.Text = "PABCSoft - NeuralArt. C# implementation of Gatys's neural style algorithm.";
      this.Icon = Icon.FromHandle((new Bitmap(Assembly.GetExecutingAssembly().GetManifestResourceStream("MainIcon.jpg"))).GetHicon());
      this.ClientSize = new Size(868, 411);
      this.MaximizeBox = false;
      this.FormBorderStyle = FormBorderStyle.FixedSingle;
      //-> ContentSettings
      this.ContentSettings = new GroupBox();
      this.ContentSettings.Text = "Content";
      this.ContentSettings.Size = new Size(276, 356);
      this.ContentSettings.Top = 10;
      this.ContentSettings.Left = 10;
      //--------------------------------------------------
        //-> ContentImageBox
        this.ContentImageBox = new PictureBox();
        this.ContentImageBox.Size = new Size(256, 256);
        this.ContentImageBox.SizeMode = PictureBoxSizeMode.Zoom;
        this.ContentImageBox.Left = 10;
        this.ContentImageBox.Top = 20;
        this.ContentImageBox.BorderStyle = BorderStyle.FixedSingle;
        this.ContentSettings.Controls.Add(this.ContentImageBox);
        //-> OpenContent
        this.OpenContent = new Button();
        this.OpenContent.Text = "Open the content image";
        this.OpenContent.Size = new Size(256, 25);
        this.OpenContent.Left = 10;
        this.OpenContent.Top = 286;
        this.ContentSettings.Controls.Add(this.OpenContent);
        //-> ContentSizes
        this.ContentSizes = new ComboBox();
        this.ContentSizes.Size = new Size(256, 25);
        this.ContentSizes.Left = 10;
        this.ContentSizes.Top = 321;
        this.ContentSizes.Items.Add("QVGA(320×240)");
        this.ContentSizes.Items.Add("HVGA(480×320)");
        this.ContentSizes.Items.Add("VGA(640×480)");
        this.ContentSizes.Items.Add("SVGA(800×600)");
        this.ContentSizes.Items.Add("HD(1280×720)");
        this.ContentSizes.SelectedIndex = 1;
        this.ContentSizes.DropDownStyle = ComboBoxStyle.DropDownList;
        this.ContentSettings.Controls.Add(this.ContentSizes);
      //--------------------------------------------------
      this.Controls.Add(this.ContentSettings);
      //-> ResultSettings
      this.ResultSettings = new GroupBox();
      this.ResultSettings.Text = "Result";
      this.ResultSettings.Size = new Size(276, 356);
      this.ResultSettings.Top = 10;
      this.ResultSettings.Left = 296;
      //--------------------------------------------------
        //-> ResultImageBox
        this.ResultImageBox = new PictureBox();
        this.ResultImageBox.Size = new Size(256, 256);
        this.ResultImageBox.SizeMode = PictureBoxSizeMode.Zoom;
        this.ResultImageBox.Left = 10;
        this.ResultImageBox.Top = 20;
        this.ResultImageBox.BorderStyle = BorderStyle.FixedSingle;
        this.ResultSettings.Controls.Add(this.ResultImageBox);
        //-> SaveResult
        this.SaveResult = new Button();
        this.SaveResult.Text = "Save the result";
        this.SaveResult.Size = new Size(256, 25);
        this.SaveResult.Left = 10;
        this.SaveResult.Top = 286;
        this.ResultSettings.Controls.Add(this.SaveResult);
        //-> ResultNorm
        this.ResultNorm = new CheckBox();
        this.ResultNorm.Text = "Balanse";
        this.ResultNorm.Checked = true;
        this.ResultNorm.Size = new Size(256, 25);
        this.ResultNorm.Top = 321;
        this.ResultNorm.Left = 10;
        this.ResultSettings.Controls.Add(this.ResultNorm);
      //--------------------------------------------------
      this.Controls.Add(this.ResultSettings);
      //-> StyleSettings
      this.StyleSettings = new GroupBox();
      this.StyleSettings.Text = "Style";
      this.StyleSettings.Size = new Size(276, 356);
      this.StyleSettings.Top = 10;
      this.StyleSettings.Left = 582;
      //--------------------------------------------------
        //-> StyleImageBox
        this.StyleImageBox = new PictureBox();
        this.StyleImageBox.Size = new Size(256, 256);
        this.StyleImageBox.SizeMode = PictureBoxSizeMode.Zoom;
        this.StyleImageBox.Left = 10;
        this.StyleImageBox.Top = 20;
        this.StyleImageBox.BorderStyle = BorderStyle.FixedSingle;
        this.StyleSettings.Controls.Add(this.StyleImageBox);
        //-> OpenStyle
        this.OpenStyle = new Button();
        this.OpenStyle.Text = "Open the style image";
        this.OpenStyle.Size = new Size(256, 25);
        this.OpenStyle.Left = 10;
        this.OpenStyle.Top = 286;
        this.StyleSettings.Controls.Add(this.OpenStyle);
        //-> StyleSizes
        this.StyleSizes = new ComboBox();
        this.StyleSizes.Size = new Size(256, 25);
        this.StyleSizes.Left = 10;
        this.StyleSizes.Top = 321;
        this.StyleSizes.Items.Add("QVGA(320×240)");
        this.StyleSizes.Items.Add("HVGA(480×320)");
        this.StyleSizes.Items.Add("VGA(640×480)");
        this.StyleSizes.Items.Add("SVGA(800×600)");
        this.StyleSizes.Items.Add("HD(1280×720)");
        this.StyleSizes.SelectedIndex = 1;
        this.StyleSizes.DropDownStyle = ComboBoxStyle.DropDownList;
        this.StyleSettings.Controls.Add(this.StyleSizes);
      //--------------------------------------------------
      this.Controls.Add(this.StyleSettings);
      //-> StartProcess
      this.StartProcess = new Button();
      this.StartProcess.Text = "Start iterative process";
      this.StartProcess.Size = new Size(276, 25);
      this.StartProcess.Left = 10;
      this.StartProcess.Top = 376;
      this.Controls.Add(this.StartProcess);
      //-> Iteration
      this.Iteration = new Label();
      this.Iteration.Text = "Iteration: 0; Time: 0";
      this.Iteration.TextAlign = ContentAlignment.MiddleCenter;
      this.Iteration.Size = new Size(276, 25);
      this.Iteration.Left = 296;
      this.Iteration.Top = 376;
      this.Controls.Add(this.Iteration);
      //-> StopProcess
      this.StopProcess = new Button();
      this.StopProcess.Text = "Stop iterative process";
      this.StopProcess.Enabled = false;
      this.StopProcess.Size = new Size(276, 25);
      this.StopProcess.Left = 582;
      this.StopProcess.Top = 376;
      this.Controls.Add(this.StopProcess);
    }

  }

}