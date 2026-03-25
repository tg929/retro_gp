from rdkit import Chem, rdBase


def canonicalize_smiles(smiles):
    smiles = (smiles or "").replace(" ", "").strip()
    if not smiles:
        return None

    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def canonicalize_precursor_set(smiles_text):
    smiles_text = (smiles_text or "").replace(" ", "").strip()
    if not smiles_text:
        return None

    components = [part for part in smiles_text.split(".") if part]
    if not components:
        return None
    canonical_components = []
    for component in components:
        canonical_component = canonicalize_smiles(component)
        if canonical_component is None:
            return None
        canonical_components.append(canonical_component)

    canonical_components.sort()
    return ".".join(canonical_components)
