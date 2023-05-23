import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
import seaborn as sns

from stmol import showmol
import py3Dmol

from rdkit import Chem
from rdkit.Chem import AllChem
import tabulate


class Emoji:
    microscope = '\U0001F52C'
    test_tube = '\U0001F9EA'
    yes = '\u2705'
    no = '\u274C'
    ruler = '\U0001F4CF'
    warning = '\u26A0'
    rocket = '\U0001F680'


def make_block(smi: str) -> str:
    mol = Chem.MolFromSmiles(smi)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    mblock = Chem.MolToMolBlock(mol)
    return mblock


def render_mol(xyz: str):
    xyzview = py3Dmol.view()
    xyzview.addModel(xyz, 'mol')
    xyzview.setStyle({'stick': {}})
    xyzview.setBackgroundColor('white')
    xyzview.zoomTo()
    showmol(xyzview, height=500, width=500)
    return xyzview


def plot_3d_mol(smile: str):
    blk = make_block(smile)
    view = render_mol(blk)
    return view


def report_molecule_classification(name: str, y_truth: bool, out: Optional[float], smile: str):
    table = [
        ["Molecule:", name],
        ["BBBP:", f"{y_truth} (target) {Emoji.microscope}"],
    ]
    if out is not None:
        table.append(["Prediction:", f"{bool(out > 0.5)} {Emoji.test_tube}"])
        table.append(["Correct:", f"{Emoji.yes}" if bool(out > 0.5) == y_truth else f"{Emoji.no}"])
    print(tabulate.tabulate(table, tablefmt="heavy_grid"))

    return plot_3d_mol(smile)


def report_molecule_regression(name: str, y_truth: float, out: Optional[float], smile: str, mask=None):
    table = [
        ["Molecule:", name],
        ["exp:", f"{y_truth:.4f} (target) {Emoji.microscope}"],
    ]
    if out is not None:
        err = abs(y_truth - out)
        table.append(["Prediction:", f"{out:.4f} {Emoji.test_tube}"])
        table.append(["|err|:", f"{err:.4f} " + (f"{Emoji.ruler}" if err < 1.5 else f"{Emoji.warning}")])
    print(tabulate.tabulate(table, tablefmt="heavy_grid"))
    return plot_3d_mol(smile)


def plot_smoothed_loss(epoch_losses: np.ndarray, window_size: int = 10):
    moving_avg = np.convolve(epoch_losses, np.ones(window_size) / window_size, mode='valid')
    moving_avg = np.clip(moving_avg, 0, None)

    q1, q3 = np.percentile(epoch_losses, [25, 75])
    iqr = q3 - q1

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(moving_avg, color='#FF6F79')
    ax.fill_between(
        range(len(moving_avg)), np.clip(moving_avg - iqr, 0, None), moving_avg + iqr, alpha=0.3, color='#FF6F79'
    )

    ax.set_title('Smoothed Loss with IQR')
    ax.set_xlabel('Steps')
    ax.set_ylabel('Loss')

    plt.show()


def plot_contours(test_y_true: np.ndarray, test_y_hat: np.ndarray, r2: float, mae: float):
    plt.style.use('seaborn')

    hist, xedges, yedges = np.histogram2d(test_y_true, test_y_hat, bins=10)
    X, Y = np.meshgrid(xedges[:-1], yedges[:-1])
    Z = hist.T
    plt.contour(X, Y, Z, colors=None, levels=5, linewidths=1.5, alpha=0.7, cmap='viridis')

    plt.scatter(test_y_true, test_y_hat, alpha=0.7, edgecolors='k', linewidths=0.5)

    plt.gca().annotate(
        "$R2 = {:.2f}$\n MAE = {:.2f}".format(r2, mae),
        xy=(0.05, 0.9),
        xycoords='axes fraction',
        size=10,
        bbox=dict(boxstyle="round", fc=(1.0, 0.7, 0.7), ec="none"),
    )

    plt.xlabel("y true")
    plt.ylabel("y pred")

    plt.show()
