# some handy functions to use along widgets
from IPython.display import display, Markdown, clear_output
# widget packages
import ipywidgets as widgets# defining some widgets
import subprocess
import sys
import os

projectIdBox = widgets.Text(
       value='Paperspace project Id',
       description='Project ID', )
tokenBox = widgets.Password(description='Token:', placeholder='Paperspace token')
button = widgets.Button(description='Deploy')
shutdownButton = widgets.Button(description='Shutdown')
out = widgets.Output()
out2 = widgets.Output()
projectId=None
token=None
deploymentId=None

def on_button_clicked(_):
    global projectId
    global tokenBox
    projectId=projectIdBox.value
    token=tokenBox.value
    process = subprocess.Popen(["sh", "deploy.sh", projectId, "whisper", token],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT)
    with out:
        clear_output()
        for c in iter(lambda: process.stdout.read(1), b""):
            sys.stdout.write(c)


def on_shutdown_clicked(_):
    global projectId
    global tokenBox
    global deploymentId
    deploymentId = os.getenv("DEPLOYMENT_ID")
    process = subprocess.Popen(["gradient", "deployments", "delete", "--id", deploymentId],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT)
    with out2:
        clear_output()
        print(deploymentId)
        for c in iter(lambda: process.stdout.read(1), b""):
            sys.stdout.write(c)
        
button.on_click(on_button_clicked)
shutdownButton.on_click(on_shutdown_clicked)
def deploy():
    box = widgets.VBox([projectIdBox, tokenBox, button, out])
    return box

def shutdown():
    box = widgets.VBox([shutdownButton, out2])
    return box

    
    