-------------------------------------
ПРОБЛЕМА - os.abort() в ML worker

Текст ошибки.
Translated Report (Full Report Below)
-------------------------------------
Process:             Python [67402]
Path:                /private/tmp/*/Python.framework/Versions/3.11/Resources/Python.app/Contents/MacOS/Python
Identifier:          org.python.python
Version:             3.11.15 (3.11.15)
Code Type:           ARM-64 (Native)
Role:                Unspecified
Parent Process:      Python [67373]
Coalition:           com.trae.app [4987]
Responsible Process: Electron [43264]
User ID:             501

Date/Time:           2026-09-23 00:02:28.5379 +0200
Launch Time:         2026-09-23 00:02:26.5066 +0200
Hardware Model:      Mac15,6
OS Version:          macOS 26.6.2 (25G83)
Release Type:        User

Crash Reporter Key:  9A1233FA-0AF3-4448-8422-B27E32DF4BDF
Incident Identifier: 2240659E-F38C-441C-9397-A068B39A2609

Sleep/Wake UUID:       A57801E2-264E-43BC-B9F3-ACBE03DF5EF4

Time Awake Since Boot: 49000 seconds
Time Since Wake:       30832 seconds

System Integrity Protection: enabled

Triggered by Thread: 0, Dispatch Queue: com.apple.main-thread

Exception Type:    EXC_CRASH (SIGABRT)
Exception Codes:   0x0000000000000000, 0x0000000000000000

Termination Reason:  Namespace SIGNAL, Code 6, Abort trap: 6
Terminating Process: Python [67402]


Application Specific Information:
abort() called


Thread 0 Crashed::  Dispatch queue: com.apple.main-thread
0   libsystem_kernel.dylib        	       0x1885c05e8 __pthread_kill + 8
1   libsystem_pthread.dylib       	       0x1885fb8d8 pthread_kill + 296
2   libsystem_c.dylib             	       0x188501978 abort + 148
3   Python                        	       0x100df1360 os_abort + 12
4   Python                        	       0x100c7933c cfunction_vectorcall_NOARGS + 88
5   Python                        	       0x100d09c34 _PyEval_EvalFrameDefault + 37124
6   Python                        	       0x100d0dc70 _PyEval_Vector + 116
7   Python                        	       0x100c3197c method_vectorcall + 304
8   Python                        	       0x100c2ef64 _PyVectorcall_Call + 152
9   Python                        	       0x100d0b820 _PyEval_EvalFrameDefault + 44272
10  Python                        	       0x100cfff1c PyEval_EvalCode + 168
11  Python                        	       0x100cfc458 builtin_exec + 428
12  Python                        	       0x100c7928c cfunction_vectorcall_FASTCALL_KEYWORDS + 76
13  Python                        	       0x100d09c34 _PyEval_EvalFrameDefault + 37124
14  Python                        	       0x100d0dc70 _PyEval_Vector + 116
15  Python                        	       0x100d6e3ec pymain_run_module + 208
16  Python                        	       0x100d6df34 Py_RunMain + 648
17  Python                        	       0x100d6f0a0 Py_BytesMain + 44
18  dyld                          	       0x1882304e4 start + 6992

Thread 1:: aha.sbox.host-monitor
0   libsystem_kernel.dylib        	       0x1885bb308 __semwait_signal + 8
1   libsystem_c.dylib             	       0x188496cc0 nanosleep + 220
2   .sboxlib_9b0e3c38             	       0x100488cb4 0x100440000 + 298164
3   libsystem_pthread.dylib       	       0x1885fbc58 _pthread_start + 136
4   libsystem_pthread.dylib       	       0x1885f6c1c thread_start + 8

Thread 2:

Thread 3:


Thread 0 crashed with ARM Thread State (64-bit):
    x0: 0x0000000000000000   x1: 0x0000000000000000   x2: 0x0000000000000000   x3: 0x0000000000000000
    x4: 0xfffffffff97c6550   x5: 0x0000000000000010   x6: 0x0000000000000045   x7: 0x0000000000000002
    x8: 0x5670bb22e5feb528   x9: 0x5670bb2310fd14a8  x10: 0x0000000000000002  x11: 0x00000000fffffffd
   x12: 0x0000000000000000  x13: 0x0000000000000000  x14: 0x0000000000000000  x15: 0x0000000000000000
   x16: 0x0000000000000148  x17: 0x00000001f667b358  x18: 0x0000000000000000  x19: 0x0000000000000006
   x20: 0x0000000000000103  x21: 0x00000001f503a260  x22: 0x0000000754ee5ca2  x23: 0x00000001003af240
   x24: 0x0000000100fc35d8  x25: 0x0000000000000000  x26: 0x0000000000000000  x27: 0x00000001004005b8
   x28: 0x0000000754ee5ca4   fp: 0x000000016fb5f900   lr: 0x00000001885fb8d8
    sp: 0x000000016fb5f8e0   pc: 0x00000001885c05e8 cpsr: 0x40001000
   far: 0x0000000000000000  esr: 0x56000080 (Syscall)

Binary Images:
       0x10029c000 -        0x10029ffff org.python.python (3.11.15) <c9a76bb2-c786-3671-9110-720591aae2d8> /private/tmp/*/Python.framework/Versions/3.11/Resources/Python.app/Contents/MacOS/Python
       0x100bd0000 -        0x100e73fff org.python.python (3.11.15, (c) 2001-2023 Python Software Foundation.) <1a31b4f0-d654-3447-9e8a-44d2f87d0f7d> /opt/homebrew/*/Python.framework/Versions/3.11/Python
       0x100440000 -        0x1004e7fff .sboxlib_9b0e3c38 (*) <f7c3fec2-1c8e-3f74-9545-bb4f46cd8cb1> /private/tmp/.sboxlib_9b0e3c38
       0x10041c000 -        0x100423fff _struct.cpython-311-darwin.so (*) <266892c9-b7fd-381a-8fc1-1d10a053a790> /opt/homebrew/*/Python.framework/Versions/3.11/lib/python3.11/lib-dynload/_struct.cpython-311-darwin.so
       0x10062c000 -        0x10063ffff _pickle.cpython-311-darwin.so (*) <663322c9-f84d-3eb8-8147-f4ae0ace45af> /opt/homebrew/*/Python.framework/Versions/3.11/lib/python3.11/lib-dynload/_pickle.cpython-311-darwin.so
       0x100650000 -        0x10065ffff _socket.cpython-311-darwin.so (*) <121b4642-f800-38a7-8298-f5995dacaf94> /opt/homebrew/*/Python.framework/Versions/3.11/lib/python3.11/lib-dynload/_socket.cpython-311-darwin.so
       0x10066c000 -        0x100677fff math.cpython-311-darwin.so (*) <1f2048df-4cf5-3b58-9836-4c4daf5f5aa2> /opt/homebrew/*/Python.framework/Versions/3.11/lib/python3.11/lib-dynload/math.cpython-311-darwin.so
       0x100608000 -        0x10060ffff select.cpython-311-darwin.so (*) <f2a336e2-a775-3718-91de-ed687cd07864> /opt/homebrew/*/Python.framework/Versions/3.11/lib/python3.11/lib-dynload/select.cpython-311-darwin.so
       0x100684000 -        0x10068bfff array.cpython-311-darwin.so (*) <6ca1041b-f930-390c-9852-1666852eb54d> /opt/homebrew/*/Python.framework/Versions/3.11/lib/python3.11/lib-dynload/array.cpython-311-darwin.so
       0x100698000 -        0x10069ffff zlib.cpython-311-darwin.so (*) <40e21c4c-2ef5-3961-bcc9-e2d2cbd173a2> /opt/homebrew/*/Python.framework/Versions/3.11/lib/python3.11/lib-dynload/zlib.cpython-311-darwin.so
       0x100430000 -        0x100433fff _bz2.cpython-311-darwin.so (*) <59511efc-d04a-38cf-9b74-b3b872e57f8b> /opt/homebrew/*/Python.framework/Versions/3.11/lib/python3.11/lib-dynload/_bz2.cpython-311-darwin.so
       0x100b44000 -        0x100b4bfff _lzma.cpython-311-darwin.so (*) <138c15fb-590b-3793-9a8d-55b3bd9ea9fc> /opt/homebrew/*/Python.framework/Versions/3.11/lib/python3.11/lib-dynload/_lzma.cpython-311-darwin.so
       0x100b8c000 -        0x100babfff liblzma.5.dylib (*) <43c01aff-c040-33ff-bb38-5e1ec42d9e75> /opt/homebrew/*/liblzma.5.dylib
       0x10061c000 -        0x10061ffff _bisect.cpython-311-darwin.so (*) <132969b6-8cd9-3094-acf7-0cc58f24c5e4> /opt/homebrew/*/Python.framework/Versions/3.11/lib/python3.11/lib-dynload/_bisect.cpython-311-darwin.so
       0x100b2c000 -        0x100b2ffff _random.cpython-311-darwin.so (*) <44d59af5-17b3-3700-b121-8b96e36ee903> /opt/homebrew/*/Python.framework/Versions/3.11/lib/python3.11/lib-dynload/_random.cpython-311-darwin.so
       0x100b5c000 -        0x100b5ffff _sha512.cpython-311-darwin.so (*) <65f910e4-b1fc-3f74-ae9f-584dc9563566> /opt/homebrew/*/Python.framework/Versions/3.11/lib/python3.11/lib-dynload/_sha512.cpython-311-darwin.so
       0x100b6c000 -        0x100b6ffff _multiprocessing.cpython-311-darwin.so (*) <5aa4023c-9ea6-3335-ac8c-f545b1ef27c1> /opt/homebrew/*/Python.framework/Versions/3.11/lib/python3.11/lib-dynload/_multiprocessing.cpython-311-darwin.so
       0x100b7c000 -        0x100b7ffff fcntl.cpython-311-darwin.so (*) <2411bd66-60ab-3fd6-bc62-9ff385bae5d9> /opt/homebrew/*/Python.framework/Versions/3.11/lib/python3.11/lib-dynload/fcntl.cpython-311-darwin.so
       0x100bbc000 -        0x100bbffff _posixsubprocess.cpython-311-darwin.so (*) <bb2acb34-ca4d-381b-b1e2-d2b535bdc357> /opt/homebrew/*/Python.framework/Versions/3.11/lib/python3.11/lib-dynload/_posixsubprocess.cpython-311-darwin.so
       0x1014b8000 -        0x1014bbfff _typing.cpython-311-darwin.so (*) <a9ca9a38-3075-32ad-b472-83ee925b8d00> /opt/homebrew/*/Python.framework/Versions/3.11/lib/python3.11/lib-dynload/_typing.cpython-311-darwin.so
       0x1058b4000 -        0x1058bbfff _hashlib.cpython-311-darwin.so (*) <c41c6ea4-f0f6-3129-9a71-2aafc88e6e79> /opt/homebrew/*/Python.framework/Versions/3.11/lib/python3.11/lib-dynload/_hashlib.cpython-311-darwin.so
       0x105d74000 -        0x1060b7fff libcrypto.3.dylib (*) <b5fb28db-1ac6-383c-9ddd-585812a35ce8> /opt/homebrew/*/libcrypto.3.dylib
       0x1014c8000 -        0x1014cffff _blake2.cpython-311-darwin.so (*) <3d903ea1-a2ab-38af-baa9-a85f28213162> /opt/homebrew/*/Python.framework/Versions/3.11/lib/python3.11/lib-dynload/_blake2.cpython-311-darwin.so
       0x10621c000 -        0x106507fff _multiarray_umath.cpython-311-darwin.so (*) <8993239a-0c5e-3db6-aa59-18f1e6b8154b> /Users/USER/*/_multiarray_umath.cpython-311-darwin.so
       0x1059ec000 -        0x1059fbfff _datetime.cpython-311-darwin.so (*) <6f67f95d-02fa-388e-88ce-cb79b8273226> /opt/homebrew/*/Python.framework/Versions/3.11/lib/python3.11/lib-dynload/_datetime.cpython-311-darwin.so
       0x1014dc000 -        0x1014dffff _contextvars.cpython-311-darwin.so (*) <27b30469-f067-3608-8fee-831b98eaf83f> /opt/homebrew/*/Python.framework/Versions/3.11/lib/python3.11/lib-dynload/_contextvars.cpython-311-darwin.so
       0x1059cc000 -        0x1059cffff _opcode.cpython-311-darwin.so (*) <679bab99-729c-3ca4-987e-547dc6aaf33e> /opt/homebrew/*/Python.framework/Versions/3.11/lib/python3.11/lib-dynload/_opcode.cpython-311-darwin.so
       0x105c30000 -        0x105c43fff _ctypes.cpython-311-darwin.so (*) <2bd69b14-20aa-3b01-8e53-dd8022418ac1> /opt/homebrew/*/Python.framework/Versions/3.11/lib/python3.11/lib-dynload/_ctypes.cpython-311-darwin.so
       0x105c7c000 -        0x105c93fff _umath_linalg.cpython-311-darwin.so (*) <2a4f3bcf-b527-354f-acd5-cce5ed4c3be4> /Users/USER/*/_umath_linalg.cpython-311-darwin.so
       0x105c0c000 -        0x105c13fff _json.cpython-311-darwin.so (*) <3091ede2-edc1-3731-83f7-9142f5459194> /opt/homebrew/*/Python.framework/Versions/3.11/lib/python3.11/lib-dynload/_json.cpython-311-darwin.so
       0x100410000 -        0x100413fff libtorch_global_deps.dylib (*) <326a4ca3-ef80-3637-8808-f2d035929b6c> /Users/USER/*/libtorch_global_deps.dylib
       0x105c68000 -        0x105c6bfff _C.cpython-311-darwin.so (*) <27aa06c6-619e-311c-a83c-ac2f9fea019f> /Users/USER/*/_C.cpython-311-darwin.so
       0x10851c000 -        0x109607fff libtorch_python.dylib (*) <df2e5ec4-2283-39ec-a747-8325395b22d1> /Users/USER/*/libtorch_python.dylib
       0x1059dc000 -        0x1059dffff libtorch.dylib (*) <89d4a149-007d-3163-b44e-11ea84738339> /Users/USER/*/libtorch.dylib
       0x105cbc000 -        0x105cc3fff libshm.dylib (*) <be11c6e8-7714-31bc-982b-53f990ed754a> /Users/USER/*/libshm.dylib
       0x136e04000 -        0x149abbfff libtorch_cpu.dylib (*) <4b0ec896-a6cb-3767-9ea3-778903571806> /Users/USER/*/libtorch_cpu.dylib
       0x1068cc000 -        0x10696ffff libc10.dylib (*) <acaa252d-da8c-3f75-800b-934ba970a2d9> /Users/USER/*/libc10.dylib
       0x1069e0000 -        0x106a77fff libomp.dylib (*) <e56febf1-776c-35bb-b9a1-8c978a01425c> /Users/USER/*/libomp.dylib
       0x105c54000 -        0x105c57fff _heapq.cpython-311-darwin.so (*) <0a0d91b1-1203-309e-8a9d-18ab6d43626b> /opt/homebrew/*/Python.framework/Versions/3.11/lib/python3.11/lib-dynload/_heapq.cpython-311-darwin.so
       0x105ca4000 -        0x105ca7fff grp.cpython-311-darwin.so (*) <f7f008f2-12ce-33d2-870c-30d7db90d890> /opt/homebrew/*/Python.framework/Versions/3.11/lib/python3.11/lib-dynload/grp.cpython-311-darwin.so
       0x105cd8000 -        0x105cdbfff mmap.cpython-311-darwin.so (*) <98838454-9080-3eb4-8326-cc7ddb49f158> /opt/homebrew/*/Python.framework/Versions/3.11/lib/python3.11/lib-dynload/mmap.cpython-311-darwin.so
       0x105cfc000 -        0x105d03fff binascii.cpython-311-darwin.so (*) <678ce36e-8066-39e3-b4bb-b81beaa0f4df> /opt/homebrew/*/Python.framework/Versions/3.11/lib/python3.11/lib-dynload/binascii.cpython-311-darwin.so
       0x105d10000 -        0x105d17fff _csv.cpython-311-darwin.so (*) <65af29fd-9be4-31fc-b608-afec21769cf8> /opt/homebrew/*/Python.framework/Versions/3.11/lib/python3.11/lib-dynload/_csv.cpython-311-darwin.so
       0x105ce8000 -        0x105cebfff _queue.cpython-311-darwin.so (*) <ce0ee872-9743-354e-bec5-be3c30ad73e1> /opt/homebrew/*/Python.framework/Versions/3.11/lib/python3.11/lib-dynload/_queue.cpython-311-darwin.so
       0x105d38000 -        0x105d3ffff cmath.cpython-311-darwin.so (*) <ca02bfad-3323-39a2-a1b0-9ab48ff2270d> /opt/homebrew/*/Python.framework/Versions/3.11/lib/python3.11/lib-dynload/cmath.cpython-311-darwin.so
       0x105d24000 -        0x105d27fff _uuid.cpython-311-darwin.so (*) <7622a96a-9166-3667-84c3-8b8d614ae799> /opt/homebrew/*/Python.framework/Versions/3.11/lib/python3.11/lib-dynload/_uuid.cpython-311-darwin.so
       0x107df0000 -        0x107e0bfff _ssl.cpython-311-darwin.so (*) <eb50726a-8ed4-3426-add1-52ee05ece5c3> /opt/homebrew/*/Python.framework/Versions/3.11/lib/python3.11/lib-dynload/_ssl.cpython-311-darwin.so
       0x107f04000 -        0x107f97fff libssl.3.dylib (*) <1b8ab7a7-ad95-3b41-85f2-fb21b66b6423> /opt/homebrew/*/libssl.3.dylib
       0x105d4c000 -        0x105d4ffff _scproxy.cpython-311-darwin.so (*) <c7a8b54d-60eb-37da-9808-f5d4ce5cbf3e> /opt/homebrew/*/Python.framework/Versions/3.11/lib/python3.11/lib-dynload/_scproxy.cpython-311-darwin.so
       0x1080fc000 -        0x10820bfff unicodedata.cpython-311-darwin.so (*) <162d6559-2aed-35b9-ae97-01fe6b425fea> /opt/homebrew/*/Python.framework/Versions/3.11/lib/python3.11/lib-dynload/unicodedata.cpython-311-darwin.so
       0x105d5c000 -        0x105d5ffff _posixshmem.cpython-311-darwin.so (*) <28e4a4c0-c531-306a-8cd0-b0fe2fa288ff> /opt/homebrew/*/Python.framework/Versions/3.11/lib/python3.11/lib-dynload/_posixshmem.cpython-311-darwin.so
       0x107dd4000 -        0x107ddffff _asyncio.cpython-311-darwin.so (*) <b442c596-bef7-3493-9183-f801cad6247f> /opt/homebrew/*/Python.framework/Versions/3.11/lib/python3.11/lib-dynload/_asyncio.cpython-311-darwin.so
       0x115db8000 -        0x11662ffff com.apple.AGXMetalG15X-M1 (353.14) <7e10c5df-5fde-3ebf-8b6d-2664619a558b> /System/Library/Extensions/AGXMetalG15X_M1.bundle/Contents/MacOS/AGXMetalG15X_M1
       0x1885b7000 -        0x1885f42e7 libsystem_kernel.dylib (*) <c6a4a4cb-92e6-3baf-aae0-e8306259209a> /usr/lib/system/libsystem_kernel.dylib
       0x1885f5000 -        0x188601b3b libsystem_pthread.dylib (*) <a373f0b0-9880-326a-88b4-dd8be4e33072> /usr/lib/system/libsystem_pthread.dylib
       0x188489000 -        0x18850a1e7 libsystem_c.dylib (*) <d77ceb62-aff6-3cec-ba9b-4f057fbe2eb5> /usr/lib/system/libsystem_c.dylib
       0x188210000 -        0x1882c34ff dyld (*) <74e52480-c2bd-3c8d-812d-95fe2b74a096> /usr/lib/dyld
               0x0 - 0xffffffffffffffff ??? (*) <00000000-0000-0000-0000-000000000000> ???

External Modification Summary:
  Calls made by other processes targeting this process:
    task_for_pid: 0
    thread_create: 0
    thread_set_state: 0
  Calls made by this process:
    task_for_pid: 0
    thread_create: 0
    thread_set_state: 0
  Calls made by all processes on this machine:
    task_for_pid: 0
    thread_create: 0
    thread_set_state: 0

-----------
Full Report
-----------

{"app_name":"Python","timestamp":"2026-09-23 00:02:28.00 +0200","app_version":"3.11.15","slice_uuid":"c9a76bb2-c786-3671-9110-720591aae2d8","build_version":"3.11.15","platform":1,"bundleID":"org.python.python","share_with_app_devs":0,"is_first_party":0,"bug_type":"309","os_version":"macOS 26.6.2 (25G83)","roots_installed":0,"name":"Python","incident_id":"2240659E-F38C-441C-9397-A068B39A2609"}
{
  "uptime" : 49000,
  "procRole" : "Unspecified",
  "version" : 2,
  "userID" : 501,
  "deployVersion" : 210,
  "modelCode" : "Mac15,6",
  "coalitionID" : 4987,
  "osVersion" : {
    "train" : "macOS 26.6.2",
    "build" : "25G83",
    "releaseType" : "User"
  },
  "captureTime" : "2026-09-23 00:02:28.5379 +0200",
  "codeSigningMonitor" : 2,
  "incident" : "2240659E-F38C-441C-9397-A068B39A2609",
  "pid" : 67402,
  "translated" : false,
  "cpuType" : "ARM-64",
  "procLaunch" : "2026-09-23 00:02:26.5066 +0200",
  "procStartAbsTime" : 1178393684894,
  "procExitAbsTime" : 1178442427881,
  "procName" : "Python",
  "procPath" : "\/private\/tmp\/*\/Python.framework\/Versions\/3.11\/Resources\/Python.app\/Contents\/MacOS\/Python",
  "bundleInfo" : {"CFBundleShortVersionString":"3.11.15","CFBundleVersion":"3.11.15","CFBundleIdentifier":"org.python.python"},
  "storeInfo" : {"deviceIdentifierForVendor":"DD29AFAA-7579-5080-A7D3-5B828AAB0D35","thirdParty":true},
  "parentProc" : "Python",
  "parentPid" : 67373,
  "coalitionName" : "com.trae.app",
  "crashReporterKey" : "9A1233FA-0AF3-4448-8422-B27E32DF4BDF",
  "appleIntelligenceStatus" : {"reasons":["selectedSiriLanguageIneligible","selectedSiriLanguageIneligibleInfo(ru)","selectedLanguageDoesNotMatchSelectedSiriLanguage","selectedLanguageDoesNotMatchSelectedSiriLanguageInfo(system: en-GB, siri: ru)"],"state":"unavailable"},
  "developerMode" : 1,
  "responsiblePid" : 43264,
  "responsibleProc" : "Electron",
  "codeSigningID" : "org.python.python",
  "codeSigningTeamID" : "",
  "codeSigningFlags" : 570425857,
  "codeSigningValidationCategory" : 10,
  "codeSigningTrustLevel" : 4294967295,
  "codeSigningAuxiliaryInfo" : 0,
  "instructionByteStream" : {"beforePC":"fyMD1f17v6n9AwCRCuD\/l78DAJH9e8Go\/w9f1sADX9YQKYDSARAA1A==","atPC":"AwEAVH8jA9X9e7+p\/QMAkf\/f\/5e\/AwCR\/XvBqP8PX9bAA1\/WcAqA0g=="},
  "bootSessionUUID" : "A91B328B-13E9-47FE-A34C-20B730378470",
  "wakeTime" : 30832,
  "sleepWakeUUID" : "A57801E2-264E-43BC-B9F3-ACBE03DF5EF4",
  "sip" : "enabled",
  "exception" : {"codes":"0x0000000000000000, 0x0000000000000000","rawCodes":[0,0],"type":"EXC_CRASH","signal":"SIGABRT"},
  "termination" : {"flags":0,"code":6,"namespace":"SIGNAL","indicator":"Abort trap: 6","byProc":"Python","byPid":67402},
  "asi" : {"libsystem_c.dylib":["abort() called"]},
  "extMods" : {"caller":{"thread_create":0,"thread_set_state":0,"task_for_pid":0},"system":{"thread_create":0,"thread_set_state":0,"task_for_pid":0},"targeted":{"thread_create":0,"thread_set_state":0,"task_for_pid":0},"warnings":0},
  "faultingThread" : 0,
  "threads" : [{"triggered":true,"id":1283154,"threadState":{"x":[{"value":0},{"value":0},{"value":0},{"value":0},{"value":18446744073600263504},{"value":16},{"value":69},{"value":2},{"value":6228684043215353128},{"value":6228684043936666792},{"value":2},{"value":4294967293},{"value":0},{"value":0},{"value":0},{"value":0},{"value":328},{"value":8428958552},{"value":0},{"value":6},{"value":259},{"value":8405623392,"symbolLocation":224,"symbol":"_main_thread"},{"value":31489678498},{"value":4298830400},{"value":4311496152,"symbolLocation":166384,"symbol":"_PyRuntime"},{"value":0},{"value":0},{"value":4299163064},{"value":31489678500}],"flavor":"ARM_THREAD_STATE64","lr":{"value":6582941912},"cpsr":{"value":1073745920},"fp":{"value":6169164032},"sp":{"value":6169164000},"esr":{"value":1442840704,"description":"(Syscall)"},"pc":{"value":6582699496,"matchesCrashFrame":1},"far":{"value":0}},"queue":"com.apple.main-thread","frames":[{"imageOffset":38376,"symbol":"__pthread_kill","symbolLocation":8,"imageIndex":53},{"imageOffset":26840,"symbol":"pthread_kill","symbolLocation":296,"imageIndex":54},{"imageOffset":493944,"symbol":"abort","symbolLocation":148,"imageIndex":55},{"imageOffset":2233184,"symbol":"os_abort","symbolLocation":12,"imageIndex":1},{"imageOffset":693052,"symbol":"cfunction_vectorcall_NOARGS","symbolLocation":88,"imageIndex":1},{"imageOffset":1285172,"symbol":"_PyEval_EvalFrameDefault","symbolLocation":37124,"imageIndex":1},{"imageOffset":1301616,"symbol":"_PyEval_Vector","symbolLocation":116,"imageIndex":1},{"imageOffset":399740,"symbol":"method_vectorcall","symbolLocation":304,"imageIndex":1},{"imageOffset":388964,"symbol":"_PyVectorcall_Call","symbolLocation":152,"imageIndex":1},{"imageOffset":1292320,"symbol":"_PyEval_EvalFrameDefault","symbolLocation":44272,"imageIndex":1},{"imageOffset":1244956,"symbol":"PyEval_EvalCode","symbolLocation":168,"imageIndex":1},{"imageOffset":1229912,"symbol":"builtin_exec","symbolLocation":428,"imageIndex":1},{"imageOffset":692876,"symbol":"cfunction_vectorcall_FASTCALL_KEYWORDS","symbolLocation":76,"imageIndex":1},{"imageOffset":1285172,"symbol":"_PyEval_EvalFrameDefault","symbolLocation":37124,"imageIndex":1},{"imageOffset":1301616,"symbol":"_PyEval_Vector","symbolLocation":116,"imageIndex":1},{"imageOffset":1696748,"symbol":"pymain_run_module","symbolLocation":208,"imageIndex":1},{"imageOffset":1695540,"symbol":"Py_RunMain","symbolLocation":648,"imageIndex":1},{"imageOffset":1700000,"symbol":"Py_BytesMain","symbolLocation":44,"imageIndex":1},{"imageOffset":132324,"symbol":"start","symbolLocation":6992,"imageIndex":56}]},{"id":1283155,"name":"aha.sbox.host-monitor","threadState":{"x":[{"value":4},{"value":0},{"value":1},{"value":1},{"value":1},{"value":0},{"value":52},{"value":0},{"value":8405664984,"symbolLocation":0,"symbol":"clock_sem"},{"value":16387},{"value":17},{"value":2},{"value":0},{"value":0},{"value":256},{"value":0},{"value":334},{"value":8428958600},{"value":0},{"value":6169735056},{"value":6169735056},{"value":4300144640},{"value":35},{"value":0},{"value":0},{"value":0},{"value":0},{"value":0},{"value":0}],"flavor":"ARM_THREAD_STATE64","lr":{"value":6581480640},"cpsr":{"value":1610616832},"fp":{"value":6169734992},"sp":{"value":6169734944},"esr":{"value":1442840704,"description":"(Syscall)"},"pc":{"value":6582678280},"far":{"value":0}},"frames":[{"imageOffset":17160,"symbol":"__semwait_signal","symbolLocation":8,"imageIndex":53},{"imageOffset":56512,"symbol":"nanosleep","symbolLocation":220,"imageIndex":55},{"imageOffset":298164,"imageIndex":2},{"imageOffset":27736,"symbol":"_pthread_start","symbolLocation":136,"imageIndex":54},{"imageOffset":7196,"symbol":"thread_start","symbolLocation":8,"imageIndex":54}]},{"id":1283185,"frames":[],"threadState":{"x":[{"value":6170308608},{"value":8195},{"value":6169772032},{"value":0},{"value":409603},{"value":18446744073709551615},{"value":0},{"value":0},{"value":0},{"value":0},{"value":0},{"value":0},{"value":0},{"value":0},{"value":0},{"value":0},{"value":0},{"value":0},{"value":0},{"value":0},{"value":0},{"value":0},{"value":0},{"value":0},{"value":0},{"value":0},{"value":0},{"value":0},{"value":0}],"flavor":"ARM_THREAD_STATE64","lr":{"value":0},"cpsr":{"value":4096},"fp":{"value":0},"sp":{"value":6170308608},"esr":{"value":1442840704,"description":"(Syscall)"},"pc":{"value":6582922248},"far":{"value":0}}},{"id":1283186,"frames":[],"threadState":{"x":[{"value":6170882048},{"value":9219},{"value":6170345472},{"value":0},{"value":409603},{"value":18446744073709551615},{"value":0},{"value":0},{"value":0},{"value":0},{"value":0},{"value":0},{"value":0},{"value":0},{"value":0},{"value":0},{"value":0},{"value":0},{"value":0},{"value":0},{"value":0},{"value":0},{"value":0},{"value":0},{"value":0},{"value":0},{"value":0},{"value":0},{"value":0}],"flavor":"ARM_THREAD_STATE64","lr":{"value":0},"cpsr":{"value":4096},"fp":{"value":0},"sp":{"value":6170882048},"esr":{"value":1442840704,"description":"(Syscall)"},"pc":{"value":6582922248},"far":{"value":0}}}],
  "usedImages" : [
  {
    "source" : "P",
    "arch" : "arm64",
    "base" : 4297703424,
    "CFBundleShortVersionString" : "3.11.15",
    "CFBundleIdentifier" : "org.python.python",
    "size" : 16384,
    "uuid" : "c9a76bb2-c786-3671-9110-720591aae2d8",
    "path" : "\/private\/tmp\/*\/Python.framework\/Versions\/3.11\/Resources\/Python.app\/Contents\/MacOS\/Python",
    "name" : "Python",
    "CFBundleVersion" : "3.11.15"
  },
  {
    "source" : "P",
    "arch" : "arm64",
    "base" : 4307353600,
    "CFBundleShortVersionString" : "3.11.15, (c) 2001-2023 Python Software Foundation.",
    "CFBundleIdentifier" : "org.python.python",
    "size" : 2768896,
    "uuid" : "1a31b4f0-d654-3447-9e8a-44d2f87d0f7d",
    "path" : "\/opt\/homebrew\/*\/Python.framework\/Versions\/3.11\/Python",
    "name" : "Python",
    "CFBundleVersion" : "3.11.15"
  },
  {
    "source" : "P",
    "arch" : "arm64",
    "base" : 4299423744,
    "size" : 688128,
    "uuid" : "f7c3fec2-1c8e-3f74-9545-bb4f46cd8cb1",
    "path" : "\/private\/tmp\/.sboxlib_9b0e3c38",
    "name" : ".sboxlib_9b0e3c38"
  },
  {
    "source" : "P",
    "arch" : "arm64",
    "base" : 4299276288,
    "size" : 32768,
    "uuid" : "266892c9-b7fd-381a-8fc1-1d10a053a790",
    "path" : "\/opt\/homebrew\/*\/Python.framework\/Versions\/3.11\/lib\/python3.11\/lib-dynload\/_struct.cpython-311-darwin.so",
    "name" : "_struct.cpython-311-darwin.so"
  },
  {
    "source" : "P",
    "arch" : "arm64",
    "base" : 4301438976,
    "size" : 81920,
    "uuid" : "663322c9-f84d-3eb8-8147-f4ae0ace45af",
    "path" : "\/opt\/homebrew\/*\/Python.framework\/Versions\/3.11\/lib\/python3.11\/lib-dynload\/_pickle.cpython-311-darwin.so",
    "name" : "_pickle.cpython-311-darwin.so"
  },
  {
    "source" : "P",
    "arch" : "arm64",
    "base" : 4301586432,
    "size" : 65536,
    "uuid" : "121b4642-f800-38a7-8298-f5995dacaf94",
    "path" : "\/opt\/homebrew\/*\/Python.framework\/Versions\/3.11\/lib\/python3.11\/lib-dynload\/_socket.cpython-311-darwin.so",
    "name" : "_socket.cpython-311-darwin.so"
  },
  {
    "source" : "P",
    "arch" : "arm64",
    "base" : 4301701120,
    "size" : 49152,
    "uuid" : "1f2048df-4cf5-3b58-9836-4c4daf5f5aa2",
    "path" : "\/opt\/homebrew\/*\/Python.framework\/Versions\/3.11\/lib\/python3.11\/lib-dynload\/math.cpython-311-darwin.so",
    "name" : "math.cpython-311-darwin.so"
  },
  {
    "source" : "P",
    "arch" : "arm64",
    "base" : 4301291520,
    "size" : 32768,
    "uuid" : "f2a336e2-a775-3718-91de-ed687cd07864",
    "path" : "\/opt\/homebrew\/*\/Python.framework\/Versions\/3.11\/lib\/python3.11\/lib-dynload\/select.cpython-311-darwin.so",
    "name" : "select.cpython-311-darwin.so"
  },
  {
    "source" : "P",
    "arch" : "arm64",
    "base" : 4301799424,
    "size" : 32768,
    "uuid" : "6ca1041b-f930-390c-9852-1666852eb54d",
    "path" : "\/opt\/homebrew\/*\/Python.framework\/Versions\/3.11\/lib\/python3.11\/lib-dynload\/array.cpython-311-darwin.so",
    "name" : "array.cpython-311-darwin.so"
  },
  {
    "source" : "P",
    "arch" : "arm64",
    "base" : 4301881344,
    "size" : 32768,
    "uuid" : "40e21c4c-2ef5-3961-bcc9-e2d2cbd173a2",
    "path" : "\/opt\/homebrew\/*\/Python.framework\/Versions\/3.11\/lib\/python3.11\/lib-dynload\/zlib.cpython-311-darwin.so",
    "name" : "zlib.cpython-311-darwin.so"
  },
  {
    "source" : "P",
    "arch" : "arm64",
    "base" : 4299358208,
    "size" : 16384,
    "uuid" : "59511efc-d04a-38cf-9b74-b3b872e57f8b",
    "path" : "\/opt\/homebrew\/*\/Python.framework\/Versions\/3.11\/lib\/python3.11\/lib-dynload\/_bz2.cpython-311-darwin.so",
    "name" : "_bz2.cpython-311-darwin.so"
  },
  {
    "source" : "P",
    "arch" : "arm64",
    "base" : 4306780160,
    "size" : 32768,
    "uuid" : "138c15fb-590b-3793-9a8d-55b3bd9ea9fc",
    "path" : "\/opt\/homebrew\/*\/Python.framework\/Versions\/3.11\/lib\/python3.11\/lib-dynload\/_lzma.cpython-311-darwin.so",
    "name" : "_lzma.cpython-311-darwin.so"
  },
  {
    "source" : "P",
    "arch" : "arm64",
    "base" : 4307075072,
    "size" : 131072,
    "uuid" : "43c01aff-c040-33ff-bb38-5e1ec42d9e75",
    "path" : "\/opt\/homebrew\/*\/liblzma.5.dylib",
    "name" : "liblzma.5.dylib"
  },
  {
    "source" : "P",
    "arch" : "arm64",
    "base" : 4301373440,
    "size" : 16384,
    "uuid" : "132969b6-8cd9-3094-acf7-0cc58f24c5e4",
    "path" : "\/opt\/homebrew\/*\/Python.framework\/Versions\/3.11\/lib\/python3.11\/lib-dynload\/_bisect.cpython-311-darwin.so",
    "name" : "_bisect.cpython-311-darwin.so"
  },
  {
    "source" : "P",
    "arch" : "arm64",
    "base" : 4306681856,
    "size" : 16384,
    "uuid" : "44d59af5-17b3-3700-b121-8b96e36ee903",
    "path" : "\/opt\/homebrew\/*\/Python.framework\/Versions\/3.11\/lib\/python3.11\/lib-dynload\/_random.cpython-311-darwin.so",
    "name" : "_random.cpython-311-darwin.so"
  },
  {
    "source" : "P",
    "arch" : "arm64",
    "base" : 4306878464,
    "size" : 16384,
    "uuid" : "65f910e4-b1fc-3f74-ae9f-584dc9563566",
    "path" : "\/opt\/homebrew\/*\/Python.framework\/Versions\/3.11\/lib\/python3.11\/lib-dynload\/_sha512.cpython-311-darwin.so",
    "name" : "_sha512.cpython-311-darwin.so"
  },
  {
    "source" : "P",
    "arch" : "arm64",
    "base" : 4306944000,
    "size" : 16384,
    "uuid" : "5aa4023c-9ea6-3335-ac8c-f545b1ef27c1",
    "path" : "\/opt\/homebrew\/*\/Python.framework\/Versions\/3.11\/lib\/python3.11\/lib-dynload\/_multiprocessing.cpython-311-darwin.so",
    "name" : "_multiprocessing.cpython-311-darwin.so"
  },
  {
    "source" : "P",
    "arch" : "arm64",
    "base" : 4307009536,
    "size" : 16384,
    "uuid" : "2411bd66-60ab-3fd6-bc62-9ff385bae5d9",
    "path" : "\/opt\/homebrew\/*\/Python.framework\/Versions\/3.11\/lib\/python3.11\/lib-dynload\/fcntl.cpython-311-darwin.so",
    "name" : "fcntl.cpython-311-darwin.so"
  },
  {
    "source" : "P",
    "arch" : "arm64",
    "base" : 4307271680,
    "size" : 16384,
    "uuid" : "bb2acb34-ca4d-381b-b1e2-d2b535bdc357",
    "path" : "\/opt\/homebrew\/*\/Python.framework\/Versions\/3.11\/lib\/python3.11\/lib-dynload\/_posixsubprocess.cpython-311-darwin.so",
    "name" : "_posixsubprocess.cpython-311-darwin.so"
  },
  {
    "source" : "P",
    "arch" : "arm64",
    "base" : 4316692480,
    "size" : 16384,
    "uuid" : "a9ca9a38-3075-32ad-b472-83ee925b8d00",
    "path" : "\/opt\/homebrew\/*\/Python.framework\/Versions\/3.11\/lib\/python3.11\/lib-dynload\/_typing.cpython-311-darwin.so",
    "name" : "_typing.cpython-311-darwin.so"
  },
  {
    "source" : "P",
    "arch" : "arm64",
    "base" : 4387979264,
    "size" : 32768,
    "uuid" : "c41c6ea4-f0f6-3129-9a71-2aafc88e6e79",
    "path" : "\/opt\/homebrew\/*\/Python.framework\/Versions\/3.11\/lib\/python3.11\/lib-dynload\/_hashlib.cpython-311-darwin.so",
    "name" : "_hashlib.cpython-311-darwin.so"
  },
  {
    "source" : "P",
    "arch" : "arm64",
    "base" : 4392960000,
    "size" : 3424256,
    "uuid" : "b5fb28db-1ac6-383c-9ddd-585812a35ce8",
    "path" : "\/opt\/homebrew\/*\/libcrypto.3.dylib",
    "name" : "libcrypto.3.dylib"
  },
  {
    "source" : "P",
    "arch" : "arm64",
    "base" : 4316758016,
    "size" : 32768,
    "uuid" : "3d903ea1-a2ab-38af-baa9-a85f28213162",
    "path" : "\/opt\/homebrew\/*\/Python.framework\/Versions\/3.11\/lib\/python3.11\/lib-dynload\/_blake2.cpython-311-darwin.so",
    "name" : "_blake2.cpython-311-darwin.so"
  },
  {
    "source" : "P",
    "arch" : "arm64",
    "base" : 4397842432,
    "size" : 3063808,
    "uuid" : "8993239a-0c5e-3db6-aa59-18f1e6b8154b",
    "path" : "\/Users\/USER\/*\/_multiarray_umath.cpython-311-darwin.so",
    "name" : "_multiarray_umath.cpython-311-darwin.so"
  },
  {
    "source" : "P",
    "arch" : "arm64",
    "base" : 4389257216,
    "size" : 65536,
    "uuid" : "6f67f95d-02fa-388e-88ce-cb79b8273226",
    "path" : "\/opt\/homebrew\/*\/Python.framework\/Versions\/3.11\/lib\/python3.11\/lib-dynload\/_datetime.cpython-311-darwin.so",
    "name" : "_datetime.cpython-311-darwin.so"
  },
  {
    "source" : "P",
    "arch" : "arm64",
    "base" : 4316839936,
    "size" : 16384,
    "uuid" : "27b30469-f067-3608-8fee-831b98eaf83f",
    "path" : "\/opt\/homebrew\/*\/Python.framework\/Versions\/3.11\/lib\/python3.11\/lib-dynload\/_contextvars.cpython-311-darwin.so",
    "name" : "_contextvars.cpython-311-darwin.so"
  },
  {
    "source" : "P",
    "arch" : "arm64",
    "base" : 4389126144,
    "size" : 16384,
    "uuid" : "679bab99-729c-3ca4-987e-547dc6aaf33e",
    "path" : "\/opt\/homebrew\/*\/Python.framework\/Versions\/3.11\/lib\/python3.11\/lib-dynload\/_opcode.cpython-311-darwin.so",
    "name" : "_opcode.cpython-311-darwin.so"
  },
  {
    "source" : "P",
    "arch" : "arm64",
    "base" : 4391632896,
    "size" : 81920,
    "uuid" : "2bd69b14-20aa-3b01-8e53-dd8022418ac1",
    "path" : "\/opt\/homebrew\/*\/Python.framework\/Versions\/3.11\/lib\/python3.11\/lib-dynload\/_ctypes.cpython-311-darwin.so",
    "name" : "_ctypes.cpython-311-darwin.so"
  },
  {
    "source" : "P",
    "arch" : "arm64",
    "base" : 4391944192,
    "size" : 98304,
    "uuid" : "2a4f3bcf-b527-354f-acd5-cce5ed4c3be4",
    "path" : "\/Users\/USER\/*\/_umath_linalg.cpython-311-darwin.so",
    "name" : "_umath_linalg.cpython-311-darwin.so"
  },
  {
    "source" : "P",
    "arch" : "arm64",
    "base" : 4391485440,
    "size" : 32768,
    "uuid" : "3091ede2-edc1-3731-83f7-9142f5459194",
    "path" : "\/opt\/homebrew\/*\/Python.framework\/Versions\/3.11\/lib\/python3.11\/lib-dynload\/_json.cpython-311-darwin.so",
    "name" : "_json.cpython-311-darwin.so"
  },
  {
    "source" : "P",
    "arch" : "arm64",
    "base" : 4299227136,
    "size" : 16384,
    "uuid" : "326a4ca3-ef80-3637-8808-f2d035929b6c",
    "path" : "\/Users\/USER\/*\/libtorch_global_deps.dylib",
    "name" : "libtorch_global_deps.dylib"
  },
  {
    "source" : "P",
    "arch" : "arm64",
    "base" : 4391862272,
    "size" : 16384,
    "uuid" : "27aa06c6-619e-311c-a83c-ac2f9fea019f",
    "path" : "\/Users\/USER\/*\/_C.cpython-311-darwin.so",
    "name" : "_C.cpython-311-darwin.so"
  },
  {
    "source" : "P",
    "arch" : "arm64",
    "base" : 4434542592,
    "size" : 17743872,
    "uuid" : "df2e5ec4-2283-39ec-a747-8325395b22d1",
    "path" : "\/Users\/USER\/*\/libtorch_python.dylib",
    "name" : "libtorch_python.dylib"
  },
  {
    "source" : "P",
    "arch" : "arm64",
    "base" : 4389191680,
    "size" : 16384,
    "uuid" : "89d4a149-007d-3163-b44e-11ea84738339",
    "path" : "\/Users\/USER\/*\/libtorch.dylib",
    "name" : "libtorch.dylib"
  },
  {
    "source" : "P",
    "arch" : "arm64",
    "base" : 4392206336,
    "size" : 32768,
    "uuid" : "be11c6e8-7714-31bc-982b-53f990ed754a",
    "path" : "\/Users\/USER\/*\/libshm.dylib",
    "name" : "libshm.dylib"
  },
  {
    "source" : "P",
    "arch" : "arm64",
    "base" : 5215633408,
    "size" : 315326464,
    "uuid" : "4b0ec896-a6cb-3767-9ea3-778903571806",
    "path" : "\/Users\/USER\/*\/libtorch_cpu.dylib",
    "name" : "libtorch_cpu.dylib"
  },
  {
    "source" : "P",
    "arch" : "arm64",
    "base" : 4404854784,
    "size" : 671744,
    "uuid" : "acaa252d-da8c-3f75-800b-934ba970a2d9",
    "path" : "\/Users\/USER\/*\/libc10.dylib",
    "name" : "libc10.dylib"
  },
  {
    "source" : "P",
    "arch" : "arm64",
    "base" : 4405985280,
    "size" : 622592,
    "uuid" : "e56febf1-776c-35bb-b9a1-8c978a01425c",
    "path" : "\/Users\/USER\/*\/libomp.dylib",
    "name" : "libomp.dylib"
  },
  {
    "source" : "P",
    "arch" : "arm64",
    "base" : 4391780352,
    "size" : 16384,
    "uuid" : "0a0d91b1-1203-309e-8a9d-18ab6d43626b",
    "path" : "\/opt\/homebrew\/*\/Python.framework\/Versions\/3.11\/lib\/python3.11\/lib-dynload\/_heapq.cpython-311-darwin.so",
    "name" : "_heapq.cpython-311-darwin.so"
  },
  {
    "source" : "P",
    "arch" : "arm64",
    "base" : 4392108032,
    "size" : 16384,
    "uuid" : "f7f008f2-12ce-33d2-870c-30d7db90d890",
    "path" : "\/opt\/homebrew\/*\/Python.framework\/Versions\/3.11\/lib\/python3.11\/lib-dynload\/grp.cpython-311-darwin.so",
    "name" : "grp.cpython-311-darwin.so"
  },
  {
    "source" : "P",
    "arch" : "arm64",
    "base" : 4392321024,
    "size" : 16384,
    "uuid" : "98838454-9080-3eb4-8326-cc7ddb49f158",
    "path" : "\/opt\/homebrew\/*\/Python.framework\/Versions\/3.11\/lib\/python3.11\/lib-dynload\/mmap.cpython-311-darwin.so",
    "name" : "mmap.cpython-311-darwin.so"
  },
  {
    "source" : "P",
    "arch" : "arm64",
    "base" : 4392468480,
    "size" : 32768,
    "uuid" : "678ce36e-8066-39e3-b4bb-b81beaa0f4df",
    "path" : "\/opt\/homebrew\/*\/Python.framework\/Versions\/3.11\/lib\/python3.11\/lib-dynload\/binascii.cpython-311-darwin.so",
    "name" : "binascii.cpython-311-darwin.so"
  },
  {
    "source" : "P",
    "arch" : "arm64",
    "base" : 4392550400,
    "size" : 32768,
    "uuid" : "65af29fd-9be4-31fc-b608-afec21769cf8",
    "path" : "\/opt\/homebrew\/*\/Python.framework\/Versions\/3.11\/lib\/python3.11\/lib-dynload\/_csv.cpython-311-darwin.so",
    "name" : "_csv.cpython-311-darwin.so"
  },
  {
    "source" : "P",
    "arch" : "arm64",
    "base" : 4392386560,
    "size" : 16384,
    "uuid" : "ce0ee872-9743-354e-bec5-be3c30ad73e1",
    "path" : "\/opt\/homebrew\/*\/Python.framework\/Versions\/3.11\/lib\/python3.11\/lib-dynload\/_queue.cpython-311-darwin.so",
    "name" : "_queue.cpython-311-darwin.so"
  },
  {
    "source" : "P",
    "arch" : "arm64",
    "base" : 4392714240,
    "size" : 32768,
    "uuid" : "ca02bfad-3323-39a2-a1b0-9ab48ff2270d",
    "path" : "\/opt\/homebrew\/*\/Python.framework\/Versions\/3.11\/lib\/python3.11\/lib-dynload\/cmath.cpython-311-darwin.so",
    "name" : "cmath.cpython-311-darwin.so"
  },
  {
    "source" : "P",
    "arch" : "arm64",
    "base" : 4392632320,
    "size" : 16384,
    "uuid" : "7622a96a-9166-3667-84c3-8b8d614ae799",
    "path" : "\/opt\/homebrew\/*\/Python.framework\/Versions\/3.11\/lib\/python3.11\/lib-dynload\/_uuid.cpython-311-darwin.so",
    "name" : "_uuid.cpython-311-darwin.so"
  },
  {
    "source" : "P",
    "arch" : "arm64",
    "base" : 4427022336,
    "size" : 114688,
    "uuid" : "eb50726a-8ed4-3426-add1-52ee05ece5c3",
    "path" : "\/opt\/homebrew\/*\/Python.framework\/Versions\/3.11\/lib\/python3.11\/lib-dynload\/_ssl.cpython-311-darwin.so",
    "name" : "_ssl.cpython-311-darwin.so"
  },
  {
    "source" : "P",
    "arch" : "arm64",
    "base" : 4428152832,
    "size" : 606208,
    "uuid" : "1b8ab7a7-ad95-3b41-85f2-fb21b66b6423",
    "path" : "\/opt\/homebrew\/*\/libssl.3.dylib",
    "name" : "libssl.3.dylib"
  },
  {
    "source" : "P",
    "arch" : "arm64",
    "base" : 4392796160,
    "size" : 16384,
    "uuid" : "c7a8b54d-60eb-37da-9808-f5d4ce5cbf3e",
    "path" : "\/opt\/homebrew\/*\/Python.framework\/Versions\/3.11\/lib\/python3.11\/lib-dynload\/_scproxy.cpython-311-darwin.so",
    "name" : "_scproxy.cpython-311-darwin.so"
  },
  {
    "source" : "P",
    "arch" : "arm64",
    "base" : 4430217216,
    "size" : 1114112,
    "uuid" : "162d6559-2aed-35b9-ae97-01fe6b425fea",
    "path" : "\/opt\/homebrew\/*\/Python.framework\/Versions\/3.11\/lib\/python3.11\/lib-dynload\/unicodedata.cpython-311-darwin.so",
    "name" : "unicodedata.cpython-311-darwin.so"
  },
  {
    "source" : "P",
    "arch" : "arm64",
    "base" : 4392861696,
    "size" : 16384,
    "uuid" : "28e4a4c0-c531-306a-8cd0-b0fe2fa288ff",
    "path" : "\/opt\/homebrew\/*\/Python.framework\/Versions\/3.11\/lib\/python3.11\/lib-dynload\/_posixshmem.cpython-311-darwin.so",
    "name" : "_posixshmem.cpython-311-darwin.so"
  },
  {
    "source" : "P",
    "arch" : "arm64",
    "base" : 4426907648,
    "size" : 49152,
    "uuid" : "b442c596-bef7-3493-9183-f801cad6247f",
    "path" : "\/opt\/homebrew\/*\/Python.framework\/Versions\/3.11\/lib\/python3.11\/lib-dynload\/_asyncio.cpython-311-darwin.so",
    "name" : "_asyncio.cpython-311-darwin.so"
  },
  {
    "source" : "P",
    "arch" : "arm64e",
    "base" : 4661673984,
    "CFBundleShortVersionString" : "353.14",
    "CFBundleIdentifier" : "com.apple.AGXMetalG15X-M1",
    "size" : 8880128,
    "uuid" : "7e10c5df-5fde-3ebf-8b6d-2664619a558b",
    "path" : "\/System\/Library\/Extensions\/AGXMetalG15X_M1.bundle\/Contents\/MacOS\/AGXMetalG15X_M1",
    "name" : "AGXMetalG15X_M1",
    "CFBundleVersion" : "353.14"
  },
  {
    "source" : "P",
    "arch" : "arm64e",
    "base" : 6582661120,
    "size" : 250600,
    "uuid" : "c6a4a4cb-92e6-3baf-aae0-e8306259209a",
    "path" : "\/usr\/lib\/system\/libsystem_kernel.dylib",
    "name" : "libsystem_kernel.dylib"
  },
  {
    "source" : "P",
    "arch" : "arm64e",
    "base" : 6582915072,
    "size" : 52028,
    "uuid" : "a373f0b0-9880-326a-88b4-dd8be4e33072",
    "path" : "\/usr\/lib\/system\/libsystem_pthread.dylib",
    "name" : "libsystem_pthread.dylib"
  },
  {
    "source" : "P",
    "arch" : "arm64e",
    "base" : 6581424128,
    "size" : 528872,
    "uuid" : "d77ceb62-aff6-3cec-ba9b-4f057fbe2eb5",
    "path" : "\/usr\/lib\/system\/libsystem_c.dylib",
    "name" : "libsystem_c.dylib"
  },
  {
    "source" : "P",
    "arch" : "arm64e",
    "base" : 6578831360,
    "size" : 734464,
    "uuid" : "74e52480-c2bd-3c8d-812d-95fe2b74a096",
    "path" : "\/usr\/lib\/dyld",
    "name" : "dyld"
  },
  {
    "size" : 0,
    "source" : "A",
    "base" : 0,
    "uuid" : "00000000-0000-0000-0000-000000000000"
  }
],
  "sharedCache" : {
  "base" : 6577700864,
  "size" : 6010847232,
  "uuid" : "f2e86c53-6052-388b-ba71-5a0c9b569413"
},
  "legacyInfo" : {
  "threadTriggered" : {
    "queue" : "com.apple.main-thread"
  }
},
  "logWritingSignature" : "1907cbdaaee4c788148e24e3def3608027f3b849",
  "bug_type" : "309",
  "roots_installed" : 0,
  "trmStatus" : 2048,
  "trialInfo" : {
  "rollouts" : [
    {
      "rolloutId" : "644114de41e7236e6177f9bd",
      "factorPackIds" : [

      ],
      "deploymentId" : 240000013
    },
    {
      "rolloutId" : "5f72dc58705eff005a46b3a9",
      "factorPackIds" : [

      ],
      "deploymentId" : 240000015
    }
  ],
  "experiments" : [

  ]
}
}

Model: Mac15,6, BootROM 18000.161.10, proc 12:6:6:0 processors, 18 GB, SMC 
Graphics: Apple M3 Pro, Apple M3 Pro, Built-In
Display: Color LCD, 3024 x 1964 Retina, Main, MirrorOff, Online
Memory Module: LPDDR5, Micron
AirPort: spairport_wireless_card_type_wifi (0x14E4, 0x4388), wl0: Jul 10 2026 02:04:28 version 23.50.20.2.41.51.209 FWID 01-b35f0cf5
IO80211_driverkit-1566.5 "IO80211_driverkit-1566.5" Jul 31 2026 19:08:25
AirPort: 
Bluetooth: Version (null), 0 services, 0 devices, 0 incoming serial ports
Network Service: Wi-Fi, AirPort, en0
Thunderbolt Bus: MacBook Pro, Apple Inc.
Thunderbolt Bus: MacBook Pro, Apple Inc.
Thunderbolt Bus: MacBook Pro, Apple Inc.


ЗАДАЧА - найти источник os.abort(), восстановить полный lifecycle этого 155-мс дочернего процесса и проверить все пути запуска/остановки ML worker