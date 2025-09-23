#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <chrono>
#include "Timer.h"

// -------------------------
// Performance Settings for CPU
// -------------------------
void setup_performance() {
    int num_threads = std::thread::hardware_concurrency();
    if (num_threads <= 0) num_threads = 1;

    // Nur intra-op Threads = volle CPU-Nutzung
    torch::set_num_threads(num_threads);
    // Inter-op auf 1 verhindert Oversubscription
    torch::set_num_interop_threads(1);

    // Optional auch Environment Variablen setzen (unter Windows mit _putenv_s)
    // _putenv_s("OMP_NUM_THREADS", std::to_string(num_threads).c_str());
    // _putenv_s("MKL_NUM_THREADS", std::to_string(num_threads).c_str());

    std::cout << "CPU threads: " << num_threads << std::endl;
}

// -------------------------
// Parameter
// -------------------------
const double alpha = 1e-4;
const double lx = 1.0, ly = 1.0;
const int nt = 100;
const int nx = 30, ny = 30;

// Device setup
torch::Device device = torch::kCUDA;

// -------------------------
// InputData Class
// -------------------------
class InputData {
public:
    torch::Tensor X, Y, T, Xyt;
    
    InputData(double lx, double ly, double lt, int nx, int ny, int nt) {
        auto x = torch::linspace(0, lx, nx, torch::kFloat32);
        auto y = torch::linspace(0, ly, ny, torch::kFloat32);
        auto t = torch::linspace(0, lt, nt, torch::kFloat32);
        
        auto meshgrid = torch::meshgrid({x, y, t}, "ij");
        X = meshgrid[0];
        Y = meshgrid[1];
        T = meshgrid[2];
        
        Xyt = torch::stack({X.flatten(), Y.flatten(), T.flatten()}, 1).to(torch::kFloat32);
        Xyt.set_requires_grad(true);
    }
    
    torch::Tensor get_xyt() { return Xyt; }
    torch::Tensor get_x_ic() { 
        return X.index({torch::indexing::Slice(), torch::indexing::Slice(), 0}); 
    }
    torch::Tensor get_y_ic() { 
        return Y.index({torch::indexing::Slice(), torch::indexing::Slice(), 0}); 
    }
    torch::Tensor get_t_ic() { 
        return T.index({torch::indexing::Slice(), torch::indexing::Slice(), 0}); 
    }
};

// -------------------------
// PINN Model
// -------------------------
struct PINNHeatImpl : torch::nn::Module {
    torch::nn::Sequential model;
    
    PINNHeatImpl(int hidden_size = 50, int n_hidden = 5) {
        model = torch::nn::Sequential(
            torch::nn::Linear(3, hidden_size),
            torch::nn::Tanh()
        );
        
        for(int i = 0; i < n_hidden - 1; i++) {
            model->push_back(torch::nn::Linear(hidden_size, hidden_size));
            model->push_back(torch::nn::Tanh());
        }
        
        model->push_back(torch::nn::Linear(hidden_size, 1));
        register_module("model", model);
    }
    
    torch::Tensor forward(torch::Tensor xyt) {
        return 25.0 + model->forward(xyt);
    }
};
TORCH_MODULE(PINNHeat);

// -------------------------
// Helper Functions
// -------------------------
torch::Tensor gaussian2D(torch::Tensor xyt, double lx = 1.0, double ly = 1.0, double sigma = 0.15) {
    auto XY = xyt.index({"...", torch::indexing::Slice(0, 2)});
    double cx = lx / 2.0, cy = ly / 2.0;
    auto r2 = torch::pow(XY.index({"...", 0}) - cx, 2) + torch::pow(XY.index({"...", 1}) - cy, 2);
    auto g = torch::exp(-r2 / (2.0 * sigma * sigma)).unsqueeze(1);
    return g;
}

torch::Tensor initial_func2(torch::Tensor xyt, double const_val = 25.0) {
    return torch::full_like(xyt.index({"...", 0}), const_val);
}

torch::Tensor initial_func(torch::Tensor xyt) {
    return initial_func2(xyt);
}

torch::Tensor heat_source(torch::Tensor xyt) {
    return 5000.0 * gaussian2D(xyt, 1.0, 1.0, 0.1);
}

torch::Tensor sample_interior_points(torch::Tensor xyt, int n_samples = 1024) {
    int n_total = xyt.size(0);
    int n = std::min(n_samples, n_total);
    auto idx = torch::randperm(n_total, torch::kLong).index({torch::indexing::Slice(0, n)});
    return xyt.index({idx});
}

torch::Tensor sample_interior_points_biased(torch::Tensor xyt_all, int n_samples = 1024, 
                                          double frac_near0 = 0.3, double t_eps = 0.02) {
    int n_total = xyt_all.size(0);
    int n_near = static_cast<int>(n_samples * frac_near0);
    
    auto mask_near = (xyt_all.index({"...", 2}) <= t_eps);
    auto idx_near_all = torch::nonzero(mask_near).squeeze(1);
    
    std::vector<torch::Tensor> selected_tensors;
    
    if (idx_near_all.size(0) > 0) {
        int take = std::min(n_near, static_cast<int>(idx_near_all.size(0)));
        auto perm = torch::randperm(idx_near_all.size(0), torch::kLong).index({torch::indexing::Slice(0, take)});
        auto selected_near = idx_near_all.index({perm});
        selected_tensors.push_back(selected_near);
    }
    
    int n_rest = n_samples - (selected_tensors.empty() ? 0 : selected_tensors[0].size(0));
    if (n_rest > 0) {
        auto rem = torch::randperm(n_total, torch::kLong).index({torch::indexing::Slice(0, n_rest)});
        selected_tensors.push_back(rem);
    }
    
    torch::Tensor selected_tensor;
    if (selected_tensors.size() == 1) {
        selected_tensor = selected_tensors[0];
    } else if (selected_tensors.size() == 2) {
        selected_tensor = torch::cat(selected_tensors, 0);
    } else {
        // Fallback to simple random sampling
        selected_tensor = torch::randperm(n_total, torch::kLong).index({torch::indexing::Slice(0, n_samples)});
    }
    
    selected_tensor = selected_tensor.to(xyt_all.device());
    return xyt_all.index({selected_tensor});
}

// =============================
// Optimierte Physics-Loss-Funktion
// =============================
torch::Tensor physics_loss_optimized(
    PINNHeat model,           // dein Modell
    torch::Tensor xyt,        // Eingaben (x,y,t)
    double alpha,             // Wärmeleitungskoeffizient
    torch::Tensor f = torch::Tensor() // optionale Source Terms
) {
    // sicherstellen, dass xyt Gradienten zulässt
    //xyt = xyt.clone();
    //xyt.set_requires_grad(true);
    xyt = xyt.detach().requires_grad_(true);

    // ---- 1. Forward einmal ausführen ----
    auto u = model->forward(xyt); // [N,1]

    // ---- 2. Erste Ableitungen ----
    auto ones_u = torch::ones_like(u);
    // Wir brauchen create_graph=true, retain_graph=true für 2. Ableitungen
    auto grads = torch::autograd::grad({ u }, { xyt }, { ones_u },
        /*retain_graph=*/true,
        /*create_graph=*/true)[0]; // [N,3]

    auto u_x = grads.select(1, 0).unsqueeze(1);
    auto u_y = grads.select(1, 1).unsqueeze(1);
    auto u_t = grads.select(1, 2).unsqueeze(1);

    // ---- 3. Zweite Ableitungen aus ersten Gradienten ----
    auto ones_x = torch::ones_like(u_x);
    auto grad_u_x = torch::autograd::grad({ u_x }, { xyt }, { ones_x },
        /*retain_graph=*/true,
        /*create_graph=*/false)[0];
    auto u_xx = grad_u_x.select(1, 0).unsqueeze(1);

    auto ones_y = torch::ones_like(u_y);
    auto grad_u_y = torch::autograd::grad({ u_y }, { xyt }, { ones_y },
        /*retain_graph=*/true,
        /*create_graph=*/false)[0];
    auto u_yy = grad_u_y.select(1, 1).unsqueeze(1);

    // ---- 4. Source Term f behandeln ----
    torch::Tensor f_tensor;
    if (!f.defined()) {
        f_tensor = torch::zeros_like(u);
    }
    else {
        f_tensor = (f.dim() == 1) ? f.unsqueeze(1) : f;
        f_tensor = f_tensor.to(u.device());
    }

    // ---- 5. Residual und Loss ----
    auto residual = u_t - alpha * (u_xx + u_yy) - f_tensor;
    auto loss = torch::mean(residual.square());

    // Rückgabe Loss + f_tensor
    return loss;
}


torch::Tensor boundary_loss_dirichlet(PINNHeat model, torch::Tensor xyt_boundary, 
                                     torch::Tensor u_boundary = torch::Tensor()) {
    if (!u_boundary.defined()) {
        return torch::tensor(0.0, torch::TensorOptions().device(device));
    }
    auto u_pred = model->forward(xyt_boundary);
    if (u_boundary.dim() == 1) {
        u_boundary = u_boundary.unsqueeze(1);
    }
    return torch::mean(torch::pow(u_pred - u_boundary.to(u_pred.device()), 2));
}

torch::Tensor boundary_loss_robin(PINNHeat model, torch::Tensor xyt_boundary, 
                                 torch::Tensor normal_vectors, 
                                 double a, double b, torch::Tensor c) {
    xyt_boundary = xyt_boundary.clone().detach();
    xyt_boundary.set_requires_grad(true);
    
    auto u_boundary = model->forward(xyt_boundary);
    auto grads_boundary = torch::autograd::grad({u_boundary}, {xyt_boundary}, 
                                               {torch::ones_like(u_boundary)}, 
                                               true, false, true)[0];
    
    auto du_dn = torch::sum(grads_boundary * normal_vectors, 1, true);
    return torch::mean(torch::pow(a * u_boundary + b * du_dn - c, 2));
}

std::pair<torch::Tensor, torch::Tensor> generate_boundary_points_and_normals(
    double x_min, double x_max, double y_min, double y_max, 
    double t_min, double t_max, int n_per_side = 50) {
    
    auto t_rand = torch::rand({n_per_side, 1}) * (t_max - t_min) + t_min;
    
    // Left side
    auto y_left = torch::rand({n_per_side, 1}) * (y_max - y_min) + y_min;
    auto x_left = torch::full_like(y_left, x_min);
    auto normals_left = torch::tensor({{-1.0, 0.0, 0.0}}).repeat({n_per_side, 1});
    
    // Right side
    auto y_right = torch::rand({n_per_side, 1}) * (y_max - y_min) + y_min;
    auto x_right = torch::full_like(y_right, x_max);
    auto normals_right = torch::tensor({{1.0, 0.0, 0.0}}).repeat({n_per_side, 1});
    
    // Bottom side
    auto x_bottom = torch::rand({n_per_side, 1}) * (x_max - x_min) + x_min;
    auto y_bottom = torch::full_like(x_bottom, y_min);
    auto normals_bottom = torch::tensor({{0.0, -1.0, 0.0}}).repeat({n_per_side, 1});
    
    // Top side
    auto x_top = torch::rand({n_per_side, 1}) * (x_max - x_min) + x_min;
    auto y_top = torch::full_like(x_top, y_max);
    auto normals_top = torch::tensor({{0.0, 1.0, 0.0}}).repeat({n_per_side, 1});
    
    auto xyt_left = torch::cat({x_left, y_left, t_rand}, 1);
    auto xyt_right = torch::cat({x_right, y_right, t_rand}, 1);
    auto xyt_bottom = torch::cat({x_bottom, y_bottom, t_rand}, 1);
    auto xyt_top = torch::cat({x_top, y_top, t_rand}, 1);
    
    auto xyt_boundary = torch::cat({xyt_left, xyt_right, xyt_bottom, xyt_top}, 0);
    auto normals_boundary = torch::cat({normals_left, normals_right, normals_bottom, normals_top}, 0);
    
    return std::make_pair(xyt_boundary, normals_boundary);
}

void compiler_info(){
#if defined(_MSC_VER)
    std::cout << "MSVC Compiler, Version: " << _MSC_VER << "\n";
#elif defined(__clang__)
    std::cout << "Clang Compiler, Version: "
        << __clang_major__ << "."
        << __clang_minor__ << "\n";
#elif defined(__INTEL_COMPILER)
    std::cout << "Intel Compiler (klassisch), Version: "
        << __INTEL_COMPILER << "\n";
#elif defined(__INTEL_CLANG_COMPILER)
    std::cout << "Intel oneAPI (Clang), Version: "
        << __INTEL_CLANG_COMPILER << "\n";
#elif defined(__GNUC__)
    std::cout << "GCC Compiler, Version: "
        << __GNUC__ << "."
        << __GNUC_MINOR__ << "\n";
#else
    std::cout << "Unbekannter Compiler\n";
#endif
}

int main() {
    compiler_info();
    // Setup performance optimizations for CPU
    setup_performance();
    
    // Force CPU device
    device = torch::kCPU;
    std::cout << "Using CPU device with optimizations" << std::endl;
    
    auto start_total = std::chrono::steady_clock::now();
    
    // Initialize data
    std::cout << "Initializing data..." << std::endl;
    InputData xyt(lx, ly, 1.0, nx, ny, nt);
    auto XYT_all = xyt.get_xyt().to(device);
    
    // Initialize model
    std::cout << "Initializing model..." << std::endl;
    PINNHeat model;
    model->to(device);
    
    // CPU-optimized optimizer settings
    torch::optim::Adam optimizer(model->parameters(), 
                                torch::optim::AdamOptions(1e-3)
                                .weight_decay(1e-4)
                                .eps(1e-8));
    
    // Pre-compute static tensors
    std::cout << "Pre-computing static data..." << std::endl;
    auto x_ic = xyt.get_x_ic().flatten();
    auto y_ic = xyt.get_y_ic().flatten();
    auto t_ic = xyt.get_t_ic().flatten();
    auto Xyt_ic = torch::stack({x_ic, y_ic, t_ic}, 1).to(torch::kFloat32).to(device);
    auto u_ic = initial_func(Xyt_ic).to(device);
    
    // Boundary points
    auto [xyt_boundary, normals] = generate_boundary_points_and_normals(0.0, lx, 0.0, ly, 0.0, 1.0, 100); // Reduced from 200
    xyt_boundary = xyt_boundary.to(torch::kFloat32).to(device);
    normals = normals.to(device);
    auto u_boundary_target = 0.1 * torch::ones({xyt_boundary.size(0), 1}).to(device);
    
    // Training parameters
    const double eps_time = 1e-3;
    auto Xyt_ic_eps = Xyt_ic.clone().detach();
    Xyt_ic_eps.index_put_({torch::indexing::Slice(), 2}, eps_time);
    Xyt_ic_eps = Xyt_ic_eps.to(device);
    
    const int epochs = 2000;
    const double lambda_phy = 1.0;
    const double lambda_ic = 10.0;
    const double lambda_bc = 1.0;
    const double lambda_cont = 10.0;
    
    std::cout << "Starting training..." << std::endl;
    auto start_training = std::chrono::steady_clock::now();
    Helper::Timer tim("TRAINING TIMER");
    
    // Training loop with CPU-optimized batch size
    for (int epoch = 0; epoch < epochs; epoch++) {
        optimizer.zero_grad();
        
        // Smaller batch size for CPU to avoid memory issues
        auto xyt_batch = sample_interior_points_biased(XYT_all, 4096).to(device); // Reduced from 8192
        
        // Physics loss - use optimized version
        auto loss_phy = physics_loss_optimized(model, xyt_batch, alpha, heat_source(xyt_batch));
        
        // Initial condition loss
        auto u_pred_ic = model->forward(Xyt_ic);
        auto loss_ic = torch::mean((u_pred_ic - u_ic).square());
        
        // Continuity loss
        auto u_pred_ic_eps = model->forward(Xyt_ic_eps);
        auto loss_cont = torch::mean((u_pred_ic_eps - u_ic).square());
        
        // Boundary condition loss
        auto loss_bc = boundary_loss_robin(model, xyt_boundary, normals, 0.0, 1.0, 0 * u_boundary_target);
        
        auto loss = lambda_phy * loss_phy + lambda_ic * loss_ic + lambda_bc * loss_bc;
        
        loss.backward();
        optimizer.step();
        
        if (epoch % 100 == 0 || epoch == epochs - 1) { // Less frequent output
            auto epoch_time = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(epoch_time - start_training);
            
            torch::NoGradGuard no_grad;
            auto u_pred_batch = model->forward(xyt_batch);
            auto u_min = u_pred_batch.min().item<double>();
            auto u_max = u_pred_batch.max().item<double>();
            
            auto u_bc_pred = model->forward(xyt_boundary);
            auto u_bc_mean = u_bc_pred.mean().item<double>();
            
            std::cout << "Epoch " << epoch << " [" << elapsed.count()/1000.0 << "s]:\n "
                      << "Epoch " << epoch << " [" << tim.getDuration() << "s]: "
                      << "loss=" << std::scientific << std::setprecision(3) << loss.item<double>() 
                      << ", phy=" << loss_phy.item<double>()
                      << ", bc=" << loss_bc.item<double>()
                      << ", ic=" << loss_ic.item<double>()
                      << ", cont" << loss_cont.item<double>()
                      << ", u_range=[" << std::fixed << std::setprecision(3) << u_min << "," << u_max << "]" << std::endl;
        }
    }
    tim.printDuration();
    
    auto end_training = std::chrono::steady_clock::now();
    auto training_time = std::chrono::duration_cast<std::chrono::seconds>(end_training - start_training);
    std::cout << "Training completed in " << training_time.count() << " seconds!" << std::endl;
    
    std::cout << "Training completed!" << std::endl;
    
    // Prediction for visualization
    const int n_vis = 100;
    auto x_vis = torch::linspace(0, lx, n_vis);
    auto y_vis = torch::linspace(0, ly, n_vis);
    auto t_vis = torch::linspace(0, 1.0, 100);
    auto meshgrid_vis = torch::meshgrid({x_vis, y_vis}, "ij");
    auto Xv = meshgrid_vis[0];
    auto Yv = meshgrid_vis[1];
    
    // std::vector<double> u_means;
    torch::NoGradGuard no_grad;
    
    // Platz für Ergebnisse vorab anlegen
    std::vector<double> u_means(t_vis.size(0));

    for (int i = 0; i < t_vis.size(0); i++) {
        auto tval = t_vis[i].item<double>();
        auto Xyt_vis = torch::stack({
            Xv.flatten(),
            Yv.flatten(),
            torch::full_like(Xv.flatten(), tval)
            }, 1).to(device);

        // Forward Pass (CPU Thread-Safe)
        auto u_pred = model->forward(Xyt_vis);

        // Mittelwert ausrechnen
        auto u_mean = u_pred.mean().item<double>();
        u_means[i] = u_mean; // direkter Indexzugriff thread-safe

        // Optional Logging (nur aus einem Thread)
        {
            std::cout << "Frame " << static_cast<int>(tval * 100)
                << ": u mean=" << u_mean << std::endl;
        }
    }
    
    // Calculate statistics
    auto min_mean = *std::min_element(u_means.begin(), u_means.end());
    auto max_mean = *std::max_element(u_means.begin(), u_means.end());
    
    double sum = 0.0;
    for (auto val : u_means) sum += val;
    double mean_of_means = sum / u_means.size();
    
    double sq_sum = 0.0;
    for (auto val : u_means) sq_sum += (val - mean_of_means) * (val - mean_of_means);
    double std_dev = std::sqrt(sq_sum / u_means.size());
    
    std::cout << "min of u mean=" << min_mean << std::endl;
    std::cout << "max of u mean=" << max_mean << std::endl;
    std::cout << "Standard dev of u mean=" << std_dev << std::endl;
    
    // Save model
    torch::save(model, "pinn_heat_model.pt");
    std::cout << "Model saved to pinn_heat_model.pt" << std::endl;
    
    return 0;
}