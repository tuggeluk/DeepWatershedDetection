import pandas as pa

def table_pickle():
    print("building ordered table of AP")


def table_txt(txt_path):
    with open(txt_path) as f:
      ap_line = f.readlines()

    ap_line = [x.strip() for x in ap_line]

    good_lines = []
    for x in ap_line:
        if "AP for" in x:
            good_lines.append(x)

    columns = ["Class","AP"]
    ap_data = pa.DataFrame(columns=columns)
    for x in good_lines:
        sp = x.split(" ")
        name = sp[2]
        val = float(sp[4])
        ap_data = ap_data.append(pa.DataFrame([[name,val]], columns=columns),ignore_index=True)



    return ap_data



if __name__ == '__main__':
    out_dir = "/share/DeepWatershedDetection/output"
    files_pattern = "deepscores_at0.25_pr.pk"

    #table_pickle(out_dir,files_pattern)

    table = table_txt("/Users/tugg/Desktop/final_ap_mu_025.txt")
    sorted = table.sort(["AP"], ascending=False)
    print(sorted)

    df = sorted[0:20]
    print(df[["Class","AP"]].to_latex(encoding='utf-8', escape=False, index=False))

    N = 10



    # table025 = table_txt("/Users/tugg/Desktop/ap_results025.txt")
    #
    # sorted05 = table05.sort(["Class"],ascending=False)
    # sorted025 = table025.sort(["Class"], ascending=False)
    #
    # sorted05 = table05.sort(["AP"], ascending=False)
    #
    # # sanity check
    # sum(sorted05["Class"] != sorted025["Class"])
    # sorted05["AP025"] = sorted025["AP"]
    #
    #
    # sorted05 = sorted05.append(sorted025["AP"])
    #
    # final = sorted05["Class"]
    #
    # print("merge tables")