from ._kegg import KEGG_Compound



class db_Compound(KEGG_Compound):
    def __init__(self, id:str, id_type:str) -> None:
        if id_type == 'kegg':
            KEGG_Compound.__init__(self, id)
            self.dblinks['KEGG'] = [self.entry]
            del self.entry