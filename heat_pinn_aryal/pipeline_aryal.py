def pipeline_aryal():

    import torch
    import torch.nn as nn

    device = torch.device("cuda" if torch.cuda.is_available () else "cpu")

    class PINN(nn.Module):
        def __init__(self , layers , neurons , activation=nn.Tanh()):
            super(PINN , self).__init__ ()
            self.activation = activation
            self.layers = nn.ModuleList ()
            self.layers.append(nn.Linear(3, neurons))
            for _ in range(layers - 1):
                self.layers.append(nn.Linear(neurons , neurons))
            self.layers.append(nn.Linear(neurons , 1))
        def forward(self , x, y, t):
            inputs = torch.cat([x, y, t], dim=1)
            output = inputs
            for layer in self.layers[:-1]:
                output = self.activation(layer(output))
            output = self.layers[-1](output) + 25
            return output

    def pde_loss(model , x, y, t, epsilon , f):
        u = model(x, y, t)
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x , x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y , y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
        residual = u_t - epsilon * (u_xx + u_yy) - f
        return torch.mean(residual ** 2)

    def initial_loss(model , x, y, t, u0):
        u = model(x, y, t)
        return torch.mean((u - u0) ** 2)

    def boundary_loss(model, x, y, t, length):
        u = model(x, y, t)
        x_boundary = (x <= 1e-6) | (x >= length - 1e-6)
        y_boundary = (y <= 1e-6) | (y >= length - 1e-6)
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
        create_graph=True)[0]
        u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u),
        create_graph=True)[0]
        loss_x = torch.mean(u_x[x_boundary] ** 2)
        loss_y = torch.mean(u_y[y_boundary] ** 2)
        return loss_x + loss_y

    def heat_source(x, y, center_x, center_y, radius, strength, t):
        distance = torch.sqrt((x - center_x)**2 + (y - center_y)**2)
        return torch.where(t > 0, strength * torch.exp(-distance**2 / (2 * radius**2)), torch.zeros_like(t))

    def generate_data(n_points , length , total_time):
        x = torch.rand(n_points , 1, requires_grad=True) * length
        y = torch.rand(n_points , 1, requires_grad=True) * length
        t = torch.rand(n_points , 1, requires_grad=True) * total_time
        n_boundary = n_points // 10
        x_boundary = torch.cat([torch.zeros(n_boundary , 1), torch.full((
        n_boundary , 1), length)], dim
        =0)
        y_boundary = torch.cat([torch.zeros(n_boundary , 1), torch.full((
        n_boundary , 1), length)], dim
        =0)
        t_boundary = torch.rand(2 * n_boundary , 1, requires_grad=True) * total_time
        x = torch.cat([x, x_boundary , torch.rand(2 * n_boundary , 1) * length], dim=0)
        y = torch.cat([y, torch.rand(2 * n_boundary , 1) * length ,
        y_boundary], dim=0)
        t = torch.cat([t, t_boundary , t_boundary], dim=0)
        return x.to(device), y.to(device), t.to(device)

    # Data from Aryal's thesis

    ## Learning data
    leraning_rate = 1e-3
    epochs = 3000
    
    ## NN Architecture
    hid_layers = 5
    nodes = 50

    ## Physics data
    epsilon = 1e-1 # alpha
    heat_radius = 0.1
    heat_strength = 500

    ## Geometry of squared domain
    length = length_x = length_y = 1.0

    ## weights of residuals
    weight_physics = 1.0
    weight_initial = 1.0
    weight_boundary = 1.0

    ### my value
    ## weight_physics = 20.0

    model = PINN(hid_layers,nodes).to(device)
    optimizer = torch.optim.Adam(model.parameters (), lr= leraning_rate)
        
    n_points= 1000
    total_time = 5.0

    for epoch in range(epochs):
        x, y, t = generate_data(n_points , length , total_time)
        u0 = torch.full_like(x, 25) # Initial condition (room temperature)

        
        f = heat_source(x, y, 0.5*length, 0.5*length , heat_radius, heat_strength , t)
        optimizer.zero_grad ()
        loss_physics = pde_loss(model , x, y, t, epsilon , f)
        loss_initial = initial_loss(model , x, y, torch.zeros_like(t), u0)
        loss_boundary = boundary_loss(model , x, y, t, length)
        loss = weight_physics * loss_physics + weight_initial * loss_initial + weight_boundary * loss_boundary
        loss.backward ()
        optimizer.step()

        if epoch % 100 == 0 or epoch == epochs-1:
            print(f"Epoch {epoch:5d}: total={loss.item():.6e}, "
                  f"phy={loss_physics.item():.6e}, ic={loss_initial.item():.6e}, bc={loss_boundary.item():.6e}")
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch}, aktuelle Lernrate = {current_lr:.2e}")

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    import matplotlib.animation as animation
    from matplotlib.widgets import Slider, Button

    # -------------------------
    # Abfragepunkte fürs Visualisieren
    # -------------------------
    lx=ly=length
    n_vis = 100  # Auflösung
    x_vis = torch.linspace(0, lx, n_vis)
    y_vis = torch.linspace(0, ly, n_vis)
    t_vis = torch.linspace(0, 1.0, 100)
    Xv, Yv = torch.meshgrid(x_vis, y_vis, indexing='ij')

    u_frames = [ ] 
    u_means = []
    with torch.no_grad():
        for tval in t_vis:
            # x,y bleiben Grid
            x_vis, y_vis = Xv, Yv

            # t_neu hat dieselbe Form wie Xv und überall denselben Wert tval
            t_neu = torch.full_like(Xv, tval)

            # Modell auf Flattened Input aufrufen
            u_vis = model(
                x_vis.reshape(-1, 1),
                y_vis.reshape(-1, 1),
                t_neu.reshape(-1, 1)
            )

            # zurück in Gridform
            u_vis = u_vis.reshape(n_vis, n_vis).cpu().numpy()
            u_frames.append(u_vis)

            u_mean = u_vis.mean()
            u_means.append(u_mean)
            print(f"Frame {tval*100:.0f}: u mean={u_mean:.6f}")

    print(f"min of u mean={np.min(np.array(u_means))}")
    print(f"max of u mean={np.max(np.array(u_means))}")
    print(f"Standard dev of u mean={np.std(np.array(u_means))}")

    u_frames = np.array(u_frames)  # shape (nt, nx, ny)

    # Slider animation function für ML-Lösung
    def anim_slide():
        results = u_frames
        nt_vis, nx, ny = results.shape
        fig, ax = plt.subplots(figsize=(8,6))
        plt.subplots_adjust(bottom=0.25)

        vmin, vmax = results.min(), results.max()
        cax = ax.imshow(results[0], origin='lower', extent=[0, lx, 0, ly],
                        cmap='hot', vmin=vmin, vmax=vmax)
        fig.colorbar(cax, label="Temperature")
        ax.set_title(f"ML Solution - t = 0.0000")
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
            frame = int(slider.val)
            cax.set_data(results[frame])
            ax.set_title(f"Frame = {frame}")
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

    # Run slider animation
    anim_slide()

    cmap = plt.get_cmap('hot')

    n_points_plot = 100
    n_points_plot = 100

    # Visualization
    fig, ax = plt.subplots(figsize=(8, 7))
    x_plot = np.linspace(0, length, n_points_plot)
    y_plot = np.linspace(0, length, n_points_plot)
    x_plot, y_plot = np.meshgrid(x_plot, y_plot)
    x_plot = torch.tensor(x_plot.flatten(), dtype=torch.float32).unsqueeze(1).to(device)
    y_plot = torch.tensor(y_plot.flatten(), dtype=torch.float32).unsqueeze(1).to(device)

    # Create a static colorbar
    t_plot = torch.zeros_like(x_plot).to(device)
    with torch.no_grad():
        u_plot = model(x_plot, y_plot, t_plot).cpu().numpy().reshape(n_points_plot, n_points_plot)
    im = ax.imshow(u_plot, extent=[0, length, 0, length], origin='lower', cmap=cmap, vmin=25, vmax=125, animated=True)
    fig.colorbar(im, ax=ax, label='Temperature (degC)')

    # Move the text object for displaying temperature below the plot
    temp_text = ax.text(0.5, -0.1, '', transform=ax.transAxes, ha='center', va='top', fontsize=12)

    def update(frame):
        t_plot = torch.full_like(x_plot, frame * 0.1).to(device)
        with torch.no_grad():
            u_plot = model(x_plot, y_plot, t_plot).cpu().numpy().reshape(n_points_plot, n_points_plot)
    
        im.set_array(u_plot)
        ax.set_title(f'2D Heat Conduction with Center Heat Source at t={frame * 0.1:.1f}s')
    
        center_temp = u_plot[n_points_plot//2, n_points_plot//2]
        edge_temp = (u_plot[0, 0] + u_plot[0, -1] + u_plot[-1, 0] + u_plot[-1, -1]) / 4
        temp_text.set_text(f'Center Temp: {center_temp:.2f}degC, Edge Temp: {edge_temp:.2f}degC')
    
        print(f"Time: {frame * 0.1:.1f}s, Center Temp: {center_temp:.2f}degC, Edge Temp: {edge_temp:.2f}degC")
    
        return [im, temp_text]

    ax.set_xlabel('x')
    ax.set_ylabel('y')

    # Adjust the layout to make room for the text below
    plt.subplots_adjust(bottom=0.2)

    anim = FuncAnimation(fig, update, frames=51, interval=200, blit=True)

    plt.show()