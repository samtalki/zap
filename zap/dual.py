import zap.devices as dev
from zap.devices.dual import DualInjector, DualGround, DualDCLine, DualACLine

DUAL_CLASS = {
    dev.Injector: DualInjector,
    dev.Generator: DualInjector,
    dev.Load: DualInjector,
    dev.Ground: DualGround,
    dev.DCLine: DualDCLine,
    dev.ACLine: DualACLine,
}


def dualize(devices):
    return [DUAL_CLASS[type(d)](d) for d in devices]
