class Sweeper:
    def __init__(self, remove_prefix=None):
        self.remove_prefix = remove_prefix

    def sweep(self, name, values):
        if self.remove_prefix and name.startswith(self.remove_prefix):
            name = name[len(self.remove_prefix):]
        return [{name: value} for value in values]

    def product(self, sweeps):
        if len(sweeps) == 0:
            return []
        result_sweeps = sweeps[0]
        for sweep in sweeps[1:]:
            result_sweeps = [dict(**x, **y) for x in result_sweeps for y in sweep]
        return result_sweeps

    def chainit(self, sweeps):
        return sum(sweeps, [])
