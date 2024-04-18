import zap.devices as dev
from zap.devices.dual import DualInjector, DualGround, DualDCLine, DualACLine, DualBattery

DUAL_CLASS = {
    dev.Injector: DualInjector,
    dev.Generator: DualInjector,
    dev.Load: DualInjector,
    dev.Ground: DualGround,
    dev.DCLine: DualDCLine,
    dev.ACLine: DualACLine,
    dev.Battery: DualBattery,
}


def dualize(devices: list[dev.AbstractDevice]):
    return [DUAL_CLASS[type(d)](d) for d in devices]
