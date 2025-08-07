import pyopencl as cl
for plat in cl.get_platforms():
    print("Platform:", plat.name)
    for dev in plat.get_devices():
        print("  Device:", dev.name, "| Type:", cl.device_type.to_string(dev.type))