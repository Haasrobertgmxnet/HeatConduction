import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button
    
def single_plot(u_frame, lx, ly, title, cmap='hot', isolines=False, save_path=None):
    plt.figure(figsize=(6,5))
    plt.grid(False)

    plt.imshow(u_frame, origin='lower', extent=[0, lx, 0, ly], cmap=cmap)
    plt.colorbar(label="Temperature")

    if isolines:
        ny, nx = u_frame.shape
        x = np.linspace(0, lx, nx)
        y = np.linspace(0, ly, ny)
        X, Y = np.meshgrid(x, y)

        isol_color = 'green'
        if cmap == 'hot':
            isol_color = 'white'

        contours = plt.contour(X, Y, u_frame, levels=10, colors=isol_color)
        plt.clabel(contours, inline=True, fontsize=9)

    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")

    if save_path:  # wenn Speicherpfad angegeben ist
        plt.savefig(save_path, dpi=300, bbox_inches='tight')  # Bild speichern
        plt.close()  # Figur schließen, um Speicher zu sparen
    else:
        plt.show()


def anim_slide(u_frames, lx, ly, title, cmap ='hot', label="Temperature", isolines = False):
    u_frames = np.array(u_frames)
    results = u_frames
    nt_vis, nx, ny = results.shape
    fig, ax = plt.subplots(figsize=(8,6))
    plt.subplots_adjust(bottom=0.25)
    plt.grid(False)

    vmin, vmax = results.min(), results.max()
    cax = ax.imshow(results[0], origin='lower', extent=[0, lx, 0, ly],
                    cmap=cmap, vmin=vmin, vmax=vmax)
    fig.colorbar(cax, label=label)

    contours = None
    if isolines:
        ny, nx = results[0].shape
        x = np.linspace(0, lx, nx)
        y = np.linspace(0, ly, ny)
        X, Y = np.meshgrid(x, y)

        isol_color='green'
        if cmap == 'hot':
            isol_color='white'

        contours = ax.contour(X, Y, results[0], levels=10, colors=isol_color)

    ax.set_title(f"{title}, Frame = 0")
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    # Slider
    ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
    slider = Slider(ax_slider, "Time", 0, nt_vis-1, valinit=0, valstep=1)

    # Play/Stop button
    ax_button = plt.axes([0.8, 0.02, 0.1, 0.04])
    button = Button(ax_button, 'Play', color='lightgray', hovercolor='0.85')

    playing = False

    def update_slider(val):
        nonlocal contours
        frame = int(slider.val)
        cax.set_data(results[frame])
        ax.set_title(f"{title}, Frame = {frame}")

        if isolines:
            # Alte Konturen entfernen (robust für alle Matplotlib-Versionen)
            if contours is not None:
                # Lösche alle LineCollection-Objekte, die auf den Achsen sind
                for artist in contours.collections if hasattr(contours, 'collections') else []:
                    artist.remove()
                # Falls contours.collections nicht existiert, lösche einfach alle LineCollections auf den Achsen
                if not hasattr(contours, 'collections'):
                    for artist in ax.collections:
                        artist.remove()
                contours = None

            # Neue Konturen zeichnen
            contours = ax.contour(X, Y, results[frame], levels=10, colors=isol_color)

        fig.canvas.draw_idle()

    slider.on_changed(update_slider)

    def play_animation(event):
        nonlocal playing
        playing = not playing
        if playing:
            button.label.set_text('Stop')
            run_animation()
        else:
            button.label.set_text('Play')

    button.on_clicked(play_animation)

    def run_animation():
        nonlocal playing
        for frame in range(int(slider.val)+1, nt_vis):
            if not playing:
                break
            slider.set_val(frame)
            plt.pause(0.05)
        button.label.set_text('Play')
        playing = False

    plt.show()

    
