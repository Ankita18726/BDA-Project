import os


def verify_project_outputs():

    print(
        "\n" + "=" * 70
    )

    print(
        "VERIFYING PROJECT OUTPUT FILES"
    )

    print(
        "=" * 70
    )

    expected_files = [

        "outputs/charts/"
        "popularity_distribution.png",

        "outputs/charts/"
        "correlation_heatmap.png",

        "outputs/charts/"
        "feature_importance.png",

        "outputs/charts/"
        "model_rmse_comparison.png",

        "outputs/charts/"
        "model_mae_comparison.png",

        "outputs/charts/"
        "model_r2_comparison.png",

        "outputs/results/"
        "model_comparison.csv"
    ]

    missing_files = []

    for file_path in expected_files:

        if os.path.exists(file_path):

            file_size = os.path.getsize(
                file_path
            )

            if file_size > 0:

                print(
                    f"[OK] {file_path}"
                )

            else:

                print(
                    f"[EMPTY] {file_path}"
                )

                missing_files.append(
                    file_path
                )

        else:

            print(
                f"[MISSING] {file_path}"
            )

            missing_files.append(
                file_path
            )

    print("\n" + "-" * 70)

    if not missing_files:

        print(
            "ALL REQUIRED PROJECT "
            "OUTPUTS GENERATED SUCCESSFULLY"
        )

        return True

    print(
        f"{len(missing_files)} "
        "required output(s) missing."
    )

    return False