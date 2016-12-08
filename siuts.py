import os

# List of species used in classification task. The index of the list is the label for each species
species_list = ['Parus_major', 'Coloeus_monedula', 'Corvus_cornix', 'Fringilla_coelebs',
               'Erithacus_rubecula', 'Phylloscopus_collybita', 'Turdus_merula', 'Cyanistes_caeruleus',
               'Emberiza_citrinella', 'Chloris_chloris', 'Turdus_philomelos', 'Phylloscopus_trochilus',
               'Sylvia_borin', 'Apus_apus', 'Passer_domesticus', 'Luscinia_luscinia', 'Sylvia_atricapilla',
               'Ficedula_hypoleuca', 'Sylvia_communis', 'Carpodacus_erythrinus']

# PlutoF species ID-s had to be handpicked, because the names didn't always correspond to the ones in Xeno-Canto. Each ID in this list corresponds to the species in species_list
plutoF_taxon_ids = [86560, 48932, 110936, 60814, 57887, 89499, 107910, 86555, 56209, 43289, 107914, 89514, 102321, 36397, 86608, 72325, 102319, 60307, 102323, 43434]

# xeno-canto quality A is the best
acceptable_quality = ["A", "B"]


data_dir = "data/"
xeno_dir = data_dir + "xeno_recordings/"
plutoF_dir = data_dir + "plutof_recordings/"


class Recording:
    def __init__(self, id, gen, sp, label, file_url):
        self.id = id
        self.gen = gen
        self.sp = sp
        self.label = label
        self.file_url = file_url
        
    def __repr__(self):
        return "id: {0}, name: {1}_{2}, label: {3}".format(self.id, self.gen, self.sp, self.label)
        
    def get_name(self):
        """Return the scientific name - <genus_species>"""
        return "{0}_{1}".format(self.gen, self.sp)
        
    def get_filename(self):
        """Return the filename withoud extension - <genus_species-id>"""
        return "{0}_{1}-{2}".format(self.gen, self.sp, self.id)
    
def create_dir(path):
    (dirname, _) = os.path.split(path)
    if (not os.path.isdir(dirname)):
        os.makedirs(dirname)
    

