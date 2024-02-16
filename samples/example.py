from prichard_diagram import make_prichard_plot

def main() -> None:
    fig, ax = make_prichard_plot(
        data_path="./hpc_optimization_plotter/model_1b_benchmark_results.csv",
        hardware_info_path="./hpc_optimization_plotter/hardware_info.json",
        timestep_size=1.3e-7,
        show_figure=True,
    )


if __name__ == "__main__":
    main()
