from traitlets.config import Config
from jupyter_core.command import main as jupymain
from nbconvert.exporters import HTMLExporter, PDFExporter
from nbconvert.preprocessors import TagRemovePreprocessor
import subprocess
import nbformat as nbf

def report(outputdir, meshfile, fmt='html'):

    c = Config()

    c.TagRemovePreprocessor.remove_cell_tags = ("remove_cell",)
    c.TagRemovePreprocessor.remove_all_outputs_tags = ('remove_output',)
    c.TagRemovePreprocessor.remove_input_tags = ('remove_input',)    
    c.TagRemovePreprocessor.enabled=True
    c.NotebookExporter.preprocessors = ["nbconvert.preprocessors.TagRemovePreprocessor"]

    if(fmt == 'html'):
        print('export html')
        c.HTMLExporter.preprocessors = ["nbconvert.preprocessors.TagRemovePreprocessor"]
        HTMLExporter(config=c).from_filename("report.ipynb")
    else:
        print('export pdf')
        c.PDFExporter.preprocessors = ["nbconvert.preprocessors.TagRemovePreprocessor"]
        x = PDFExporter(config=c).from_filename("report.ipynb")
        print(dir(x))
    
    #commandlist = ['jupyter', 'nbconvert', '--to', fmt, 'report.ipynb']
    #print(commandlist)
    #result = subprocess.call(commandlist)

if __name__ == '__main__':

    print('++++++++++++++++++++++++ processing')
    report('output dir', 'mesh file', 'html')
    report('output dir', 'mesh file', 'pdf')

