# some handy functions to use along widgets
from IPython.display import display, Markdown, clear_output
# widget packages
import ipywidgets as widgets# defining some widgets
import subprocess
import sys
import os

# package globals
projectIdBox = None
tokenBox = None
button = None
shutdownButton = None
out = None
out2 = None
projectId=None
token=None
deploymentId=None
deploymentName="model"

def initialize(name: str):
    global deploymentName, projectIdBox, tokenBox, button, shutdownButton, out, out2
    deploymentName = name
    projectIdBox = widgets.Text(value='Paperspace project Id', description='Project ID', )
    tokenBox = widgets.Password(description='Token:', placeholder='Paperspace token')
    button = widgets.Button(description='Deploy', disabled=False)
    shutdownButton = widgets.Button(description='Shutdown', disabled=False)
    out = widgets.Output()
    out2 = widgets.Output()
    
    button.on_click(on_button_clicked)
    shutdownButton.on_click(on_shutdown_clicked)
    


def on_button_clicked(_):
    button.disabled=True
    global projectId
    global tokenBox
    projectId=projectIdBox.value
    token=tokenBox.value
    process = subprocess.Popen(["sh", "utils/deploy.sh", projectId, deploymentName, token],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT)
    with out:
        clear_output()
        for c in iter(lambda: process.stdout.read(1), b""):
            sys.stdout.write(c)

def on_shutdown_clicked(_):
    import json
    global projectId
    global tokenBox
    global deploymentId
    deployment_info = json.load(open("deployment_info.json","r"))
    deploymentId = deployment_info["id"]
    process = subprocess.Popen(["gradient", "deployments", "delete", "--id", deploymentId],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT)
    with out2:
        clear_output()
        for c in iter(lambda: process.stdout.read(1), b""):
            sys.stdout.write(c)
        

def deploy():
    box = widgets.VBox([projectIdBox, tokenBox, button, out])
    return box

def shutdown():
    box = widgets.VBox([shutdownButton, out2])
    return box

    
    