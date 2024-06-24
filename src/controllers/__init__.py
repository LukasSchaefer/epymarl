REGISTRY = {}

from .basic_controller import BasicMAC
from .experience_sharing_controller import ExperienceSharingMAC
from .non_shared_controller import NonSharedMAC
from .maddpg_controller import MADDPGMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["exp_sharing_mac"] = ExperienceSharingMAC
REGISTRY["non_shared_mac"] = NonSharedMAC
REGISTRY["maddpg_mac"] = MADDPGMAC
