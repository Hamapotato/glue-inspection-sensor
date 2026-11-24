using System.Windows;
using GlueInspect.VmViewer.ViewModels;

namespace GlueInspect.VmViewer;

public partial class MainWindow : Window
{
    public MainWindow()
    {
        InitializeComponent();
    }

    private void FitOriginal_Click(object sender, RoutedEventArgs e) => OriginalViewer.Fit();

    private void FitCaliper_Click(object sender, RoutedEventArgs e) => CaliperViewer.Fit();
}
