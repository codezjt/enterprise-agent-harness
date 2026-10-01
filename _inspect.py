from deepagents import FilesystemPermission, FilesystemMiddleware
import inspect

print("FilesystemPermission dir:", [x for x in dir(FilesystemPermission) if not x.startswith("_")])

try:
    p = FilesystemPermission("read")
    print("str ctor works:", p, type(p))
except Exception as e:
    print("str ctor error:", e)

src = inspect.getsource(FilesystemPermission)
print("\nSOURCE:", src[:800])
