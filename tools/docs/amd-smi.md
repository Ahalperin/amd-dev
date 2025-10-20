### amd-smi util to interigate local GPUs
```shell
amd-smi --help

AMD System Management Interface | Version: 26.0.0+37d158ab | ROCm version: 7.0.0 |
Platform: Linux Guest

options:
  -h, --help          show this help message and exit

AMD-SMI Commands:
                      Descriptions:
    version           Display version information
    list              List GPU information
    static            Gets static information about the specified GPU
    firmware (ucode)  Gets firmware information about the specified GPU
    metric            Gets metric/performance information about the specified GPU
    process           Lists general process information running on the specified GPU
    event             Displays event information for the given GPU
    topology          Displays topology information of the devices
    set               Set options for devices
    reset             Reset options for devices
    monitor (dmon)    Monitor metrics for target devices
    xgmi              Displays xgmi information of the devices
    partition         Displays partition information of the devices
    ras               Retrieve RAS (CPER) entries from the driver
```

### when running amd-smi from the host

```shell
amd-smi 
+------------------------------------------------------------------------------+
| AMD-SMI 26.0.0+37d158ab      amdgpu version: 6.14.14  ROCm version: 7.0.0    |
| Platform: Linux Guest                                                        |
|-------------------------------------+----------------------------------------|
| BDF                        GPU-Name | Mem-Uti   Temp   UEC       Power-Usage |
| GPU  HIP-ID  OAM-ID  Partition-Mode | GFX-Uti    Fan               Mem-Usage |
|=====================================+========================================|
| 0000:83:00.0 AMD Instinct MI300X VF | 0 %      37 °C   0           135/750 W |
|   0       0       0        SPX/NPS1 | 0 %        N/A           285/196288 MB |
+-------------------------------------+----------------------------------------+
+------------------------------------------------------------------------------+
| Processes:                                                                   |
|  GPU        PID  Process Name          GTT_MEM  VRAM_MEM  MEM_USAGE     CU % |
|==============================================================================|
|  No running processes found                                                  |
+------------------------------------------------------------------------------+
```

```shell
amd-smi list
GPU: 0
    BDF: 0000:83:00.0
    UUID: 89ff74b5-0000-1000-80c1-fb66fcc143d2
    KFD_ID: 21947
    NODE_ID: 1
    PARTITION_ID: 0
```

### when running adm-smi from a container

```shell
amd-smi     
usage: amd-smi [-h]  ...

AMD System Management Interface | Version: 25.5.1+41065ee6 | ROCm version: 6.4.4 |
Platform: Linux Guest

options:
  -h, --help          show this help message and exit

AMD-SMI Commands:
                      Descriptions:
    version           Display version information
    list              List GPU information
    static            Gets static information about the specified GPU
    firmware (ucode)  Gets firmware information about the specified GPU
    metric            Gets metric/performance information about the specified GPU
    process           Lists general process information running on the specified GPU
    event             Displays event information for the given GPU
    set               Set options for devices
    reset             Reset options for devices
    monitor (dmon)    Monitor metrics for target devices
    xgmi              Displays xgmi information of the devices
    partition         Displays partition information of the devices
    ras               Retrieve CPER (RAS) entries from the driver
```

### Reset a specific GPU (replace N with GPU number: 0, 1, 2, etc.)

```shell
sudo rocm-smi --gpureset -d N

# Example: Reset GPU 0
sudo rocm-smi --gpureset -d 0
```

### Reload AMD Kernel Module (All GPUs)
```shell
# Unload AMD GPU driver modules
sudo modprobe -r amdgpu

# Reload the driver
sudo modprobe amdgpu
```
