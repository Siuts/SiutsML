import os
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
    
