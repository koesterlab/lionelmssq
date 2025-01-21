from mass_explanation import explain_mass
import polars as pl
import re

fragment_masses = pl.read_csv("tests/testcases/test_04/2023-05-25_SJH_20merRNASidney_ms2_02_deisotoped.tsv", separator="\t")
#fragment_masses = pl.read_csv("tests/testcases/test_03/2023-04-17_SJH_10merSidney_ms2_desiotoped.tsv",separator = "\t")

neutral_masses = fragment_masses.select(pl.col("neutral_mass")).to_series().to_list()
is_start  = []
is_end    = []
skip_mass = []
nucleotide_only_masses = []

#Inferred from Shanice's RNA file!
#DNA
label_mass_3T = 455.14912 #3' label  #y-fragments 
label_mass_5T = 635.15565 #5' label #c-fragments

#Inferred from Shanice's RNA file!
#RNA
label_mass_3T = 375.18279 #3' label  #y-fragments  375,18279 Y fragments
label_mass_5T = 537.11889 #5' label #c-fragments 537,11889

# regex for separating given sequence into nucleosides
nucleoside_re5T = re.compile(r"\d*[5T]")
nucleoside_re3T = re.compile(r"\d*[3T]")

for mass in neutral_masses:
    explained_mass = explain_mass(mass)
    
    if explained_mass.explanations != set():
        #print(explained_mass.explanations,mass)
        skip_mass.append(False)
        
        temp_list = []
        for element in explained_mass.explanations:
            temp_list.extend(element)
        temp_list = ''.join(temp_list)

        #TODO: The problem is here, for heavy masses, very multiple solutions are possible
        #Out of these solutions, its very likely that terminal ones will be selected
        #Even though they might be internal fragments actually!

        #TODO: Only output if a sequence is tagged IF all possible DP solutions of the sequence have the tag in there!
        # Can also restrict this by doing this above a threshold value! 

        #TODO: use the DP in lionelmssq!!! Ask Johannes!

        if '5T' in nucleoside_re5T.findall(temp_list):
            nucleotide_only_masses.append(mass - label_mass_5T)
            is_start.append(True)
            is_end.append(False)
        elif '3T' in nucleoside_re3T.findall(temp_list):
            nucleotide_only_masses.append(mass - label_mass_3T)
            is_end.append(True)
            is_start.append(False)
        else:
            nucleotide_only_masses.append(mass)
            is_start.append(False)
            is_end.append(False)
    else:
        nucleotide_only_masses.append(mass)
        skip_mass.append(True)
        is_start.append(False)
        is_end.append(False)

fragment_masses = fragment_masses.with_columns(pl.Series(nucleotide_only_masses).alias("observed_mass")).hstack(pl.DataFrame({"is_start": is_start, "is_end": is_end})).filter(~pl.Series(skip_mass)).filter(pl.col("neutral_mass") > 305.04129 ).sort(pl.col("observed_mass"))
#.filter(pl.col("neutral_mass") < 5500 )

fragment_masses = fragment_masses.filter(pl.col("intensity") > 0.5e6)
#For terminal fragments, keep only those with _VERY HIGH_ intensity! 

print(fragment_masses)

fragment_masses.write_csv("tests/testcases/test_04/2023-05-25_SJH_20merRNASidney_ms2_02_deisotoped_terminal_marked_mass_cutoff_high_intensity.tsv",separator="\t")
#fragment_masses.write_csv("tests/testcases/test_03/2023-04-17_SJH_10merSidney_ms2_desiotoped_terminal_marked_mass_cutoff_high_intensity.tsv",separator = "\t")
