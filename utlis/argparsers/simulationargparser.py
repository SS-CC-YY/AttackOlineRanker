import argparse

class SimulationArgumentParser(argparse.ArgumentParser):
    
    def __init__(self, description=None, set_arguments={}):
        self._description = description
        self._initial_set_arguments = set_arguments.copy()
        self._set_arguments = set_arguments
        self._initial_arguments = {}
        self._simulation_arguments = []
        self._arguments_initialized = False

        super(SimulationArgumentParser, self).__init__(description=description)

        self._sim_add_argument('-L','--length', default=1000, type=int,
                        help='Number of items to extract')
        self._sim_add_argument('-d', '--dimension', default=5, type=int,
                        help='Dimension of feature space')
        self._sim_add_argument('-T', '--time', type=int,required=True,
                        help='Time t or Number of iterations')
        self._sim_add_argument('-r', '--repeat',default=1, type=int,
                        help='Number of runs')
        self._sim_add_argument('-cm', '--click_models', type=str, required=True,
                        help='Click models to be used', nargs='+')
        self._sim_add_argument('-s', '--synthetic', default=True, type=bool,
                        help='Use synthetic item or not')
        self._sim_add_argument('-t', '--tabular', default=False, type=bool,
                        help='Use diagonal function to generate items or not')
        self._sim_add_argument('-f', '--filename', type=str,
                        help='Use the provided dataset, default is MovieLens dataset',
                        default='/nfs/stak/users/songchen/research/SttackOnlineRanker/dataset/ml_1000user_1000item.npy')
        self._sim_add_argument('-a', '--algorithms', type=str, required=True,
                        help='The algorithms that would be attacked\n UCB stands for pbm-UCB or cas-UCB, Top stands for Top ranker', nargs='+')

        self._arguments_initialized = False

    def _sim_add_argument(self, *args, **kargs):
            if 'dest' in kargs:
                name = kargs['dest']
            elif args[0][:2] == '--':
                name = args[0][2:]
            else:
                assert args[0][:1] == '-'
                name = args[0][1:]

            assert name != 'description'
            if not name in self._set_arguments:
                super(SimulationArgumentParser, self).add_argument(*args, **kargs)

            assert name not in self._simulation_arguments
            self._simulation_arguments.append(name)
    def parse_sim_args(self):
        args = vars(self.parse_args())
        sim_args = {
            'description': self._description,
        }
        for name, value in args.items():
            if name in self._simulation_arguments:
                sim_args[name] = value
        return argparse.Namespace(**sim_args)

    def parse_other_args(self, ranker_args=None, ranker=None):
        args = vars(self.parse_args())
        other_args = {}
        if ranker:
            other_args.update(
            ranker.default_ranker_parameters()
            )
        for name, value in args.items():
            if name not in self._simulation_arguments:
                other_args[name] = value
        if ranker_args:
            other_args.update(ranker_args)
        return other_args

    def parse_all_args(self, ranker_args=None, ranker=None):
        return (self.parse_sim_args(),
                self.parse_other_args(
                ranker_args = ranker_args,
                ranker = ranker))