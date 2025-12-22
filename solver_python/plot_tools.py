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


import torch
@torch.no_grad()
def animate_heatmap(
    model,
    t_vals,
    lx=1.0,
    ly=1.0,
    grid_size=80,
    cmap="hot",
    isolines=False,
    title="PINN Heat Equation Solution",
    device="cpu",    
    temp_transform=None,
):
    
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider, Button

    print("animate_heatmap: Generating frames...")

    # -------------------------------------------------
    # 1) Grid
    # -------------------------------------------------
    x = torch.linspace(0, lx, grid_size, device=device)
    y = torch.linspace(0, ly, grid_size, device=device)
    X, Y = torch.meshgrid(x, y, indexing="xy")

    XY = torch.cat([X.reshape(-1,1), Y.reshape(-1,1)], dim=1)

    # -------------------------------------------------
    # 2) Model predictions for each t
    # -------------------------------------------------
    print("2) Generating u_frames ...")
    u_frames = []
    for t in t_vals:
        t_col = torch.full((XY.shape[0], 1), float(t), device=device)
        xyt = torch.cat([XY, t_col], dim=1)
        if temp_transform is None:
            u = model(xyt)[:,0].reshape(grid_size, grid_size)
        else:
            u = temp_transform.inv_scale(model(xyt)[:,0]).reshape(grid_size, grid_size)
        # print(f"u max: {u.max().item():.4f}, min: {u.min().item():.4f}")
        u_frames.append(u.cpu().numpy())

    u_frames = np.array(u_frames)
    nt, nx, ny = u_frames.shape

    # -------------------------------------------------
    # 3) Plot Setup
    # -------------------------------------------------
    print("3) Plot setup")
    fig, ax = plt.subplots(figsize=(8,6))
    plt.subplots_adjust(bottom=0.25)
    plt.grid(False)

    vmin, vmax = u_frames.min(), u_frames.max()

    cax = ax.imshow(
        u_frames[0],
        origin="lower",
        extent=[0, lx, 0, ly],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect="auto"
    )
    fig.colorbar(cax, label="Temperature")

    # Contours
    print("Contours...")
    contours = None
    if isolines:
        x_np = np.linspace(0, lx, nx)
        y_np = np.linspace(0, ly, ny)
        X_np, Y_np = np.meshgrid(x_np, y_np)
        isol_color = "white" if cmap == "hot" else "green"
        contours = ax.contour(X_np, Y_np, u_frames[0], 10, colors=isol_color)

    ax.set_title(f"{title}, t={t_vals[0]:.3f}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # -------------------------------------------------
    # 4) Slider for time
    # -------------------------------------------------
    print("4) Creating slider")
    ax_slider = plt.axes([0.15, 0.12, 0.70, 0.03])
    slider = Slider(ax_slider, "Frame", 0, nt-1, valinit=0, valstep=1)

    # -------------------------------------------------
    # 5) Play/Stop button
    # -------------------------------------------------
    print("5) Creating Play button")
    ax_button = plt.axes([0.82, 0.02, 0.12, 0.05])
    button = Button(ax_button, "Play", color="lightgray", hovercolor="0.85")

    playing = False

    # -------------------------------------------------
    # 6) Slider update
    # -------------------------------------------------
    print("6) Activating slider")
    def update_slider(val):
        nonlocal contours
        frame = int(slider.val)

        cax.set_data(u_frames[frame])
        ax.set_title(f"{title}, t={t_vals[frame]:.3f}")

        if isolines:
            # remove old contours
            if contours is not None:
                for coll in contours.collections:
                    coll.remove()
                contours = None

            contours = ax.contour(
                X_np, Y_np, u_frames[frame], 10, 
                colors=("white" if cmap == "hot" else "green")
            )

        fig.canvas.draw_idle()

    slider.on_changed(update_slider)

    # -------------------------------------------------
    # 7) Play / Stop handler
    # -------------------------------------------------
    def play(event):
        nonlocal playing
        playing = not playing
        button.label.set_text("Stop" if playing else "Play")
        if playing:
            run_animation()

    button.on_clicked(play)

    # -------------------------------------------------
    # 8) Animation loop
    # -------------------------------------------------
    def run_animation():
        nonlocal playing
        for frame in range(int(slider.val)+1, nt):
            if not playing:
                break
            slider.set_val(frame)
            plt.pause(0.05)
        playing = False
        button.label.set_text("Play")

    # -------------------------------------------------
    # 9) Show the figure
    # -------------------------------------------------
    print("9) plt.show()")
    plt.show()