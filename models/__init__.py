# Conditional imports to avoid dependency conflicts
try:
    from .DepMamba import DepMamba
except ImportError as e:
    print(f"Warning: Could not import DepMamba: {e}")
    DepMamba = None

try:
    from .MultiModalDepDet import MultiModalDepDet
except ImportError as e:
    print(f"Warning: Could not import MultiModalDepDet: {e}")
    MultiModalDepDet = None

# DeepThinkSpeech has minimal dependencies
try:
    from .DeepThinkSpeech import BiLSTM_MHA_Dysarthria, HybridExplainableModel
except ImportError:
    pass
