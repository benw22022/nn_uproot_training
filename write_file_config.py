import glob
import os


if __name__ == "__main__":

    ntuple_dir = "/home/bewilson/NTuples"

    file_dirs = glob.glob(os.path.join(ntuple_dir, "*.root"))

    file_dict = {
                "JZ1" : "user.bewilson.tauclassifier.800036.Py8EG_A14N23LO_jetjet_JZ1WwithSW_v0_output.root",
                "Gammatautau" : "user.bewilson.TauClassifierV3.425200.Pythia8EvtGen_A14NNPDF23LO_Gammatautau_MassWeight_v0_output.root",
                "JZ2" : "user.bewilson.tauclassifier.364702.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ2WithSW_v0_output.root",
                "JZ3" : "user.bewilson.tauclassifier.364703.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ3WithSW_v0_output.root",
                "JZ4" : "user.bewilson.tauclassifier.364704.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ4WithSW_v0_output.root",
                "JZ5" : "user.bewilson.tauclassifier.364705.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ5WithSW_v0_output.root",
                "JZ6" : "user.bewilson.tauclassifier.364706.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ6WithSW_v0_output.root",
                "JZ7" : "user.bewilson.tauclassifier.364707.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ7WithSW_v0_output.root",
                "JZ8" : "user.bewilson.tauclassifier.364708.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ8WithSW_v0_output.root",
                }

    with open("file_config.yaml", 'w') as file:
        file.write("---\n")
        file.write(f"NTupleDir: {ntuple_dir}\n\n")
        
        for file_type, file_dir in file_dict.items():
            file.write(f"{file_type}:\n")
            file.write(f"\t-\tDirectory: {file_dir}\n")
            file_list = glob.glob(os.path.join(ntuple_dir, file_dir, "*.root"))
            file.write("\t-\tFiles:\n")
            for f in file_list:
                file.write(f"\t\t-\t{os.path.basename(f)}\n")
            file.write("\n\n")
