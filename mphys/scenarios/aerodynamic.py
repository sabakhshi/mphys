import os

from openmdao.utils.mpi import MPI

from mphys.core import Builder, MPhysVariables, Scenario


class ScenarioAerodynamic(Scenario):
    def initialize(self):
        """
        A class to perform a single discipline aerodynamic case.
        The Scenario will add the aerodynamic builder's precoupling subsystem,
        the coupling subsystem, and the postcoupling subsystem.
        """
        super().initialize()

        self.options.declare(
            "aero_builder",
            recordable=False,
            desc="The MPhys builder for the aerodynamic solver",
        )
        self.options.declare(
            "in_MultipointParallel",
            default=False,
            types=bool,
            desc="Set to `True` if adding this scenario inside a MultipointParallel Group.",
        )
        self.options.declare(
            "geometry_builder",
            default=None,
            recordable=False,
            desc="The optional MPhys builder for the geometry",
        )
        self.options.declare(
            "geometry_comp",
            default=None,
            recordable=False,
            desc="The optional MPhys component for the geometry (alternative in case there is no builder)",
        )
        self.options.declare(
            "ffd_file",
            default="ffd.xyz",
            types=str,
            recordable=False,
            desc="This is the ffd file for the hacked comp",
        )
        self.options.declare(
            "append_scenario_to_output_path",
            default=False,
            types=bool,
            recordable=False,
            desc="This option appends the scenario name to the output folder path. Must create directories in advance!",
        )
        self.options.declare(
            "scenario_comm_callback",
            default=None,
            recordable=False,
            desc="This option allows you to pass a call back function handle for function that takes the communicator and name for this scenario as input." \
            "Intended for redirecting the IO for MultipointParallel cases but could be used for other things potentially.",
        )

    def _mphys_scenario_setup(self):
        aero_builder: Builder = self.options["aero_builder"]
        geometry_builder: Builder = self.options["geometry_builder"]
        geometry_comp: Builder = self.options["geometry_comp"]

        if self.options["append_scenario_to_output_path"]:
            aero_builder.options["outputDirectory"] = os.path.join(aero_builder.options["outputDirectory"],self.name)

        if self.options["in_MultipointParallel"]:
            if self.options["scenario_comm_callback"] is not None:
                func = self.options["scenario_comm_callback"]
                func(self.comm,self.name)

            aero_builder.initialize(self.comm)

            if geometry_builder is not None:
                geometry_builder.initialize(self.comm)
                self.mphys_add_subsystem(
                    "mesh", aero_builder.get_mesh_coordinate_subsystem(self.name)
                )
                self.mphys_add_subsystem(
                    "geometry",
                    geometry_builder.get_mesh_coordinate_subsystem(self.name),
                )
                self.connect(
                    MPhysVariables.Aerodynamics.Surface.Mesh.COORDINATES,
                    MPhysVariables.Aerodynamics.Surface.Geometry.COORDINATES_INPUT,
                )
                self.connect(
                    MPhysVariables.Aerodynamics.Surface.Geometry.COORDINATES_OUTPUT,
                    MPhysVariables.Aerodynamics.Surface.COORDINATES,
                )
                self.connect(
                    MPhysVariables.Aerodynamics.Surface.Geometry.COORDINATES_OUTPUT,
                    MPhysVariables.Aerodynamics.Surface.COORDINATES_INITIAL,
                )
            elif geometry_comp is not None:
                # Alternative that allows for insertion of a geometry group that doesn't have a builder
                self.mphys_add_subsystem(
                    "mesh", aero_builder.get_mesh_coordinate_subsystem(self.name)
                )
                self.mphys_add_subsystem(
                    "geometry",
                    geometry_comp(file=self.options["ffd_file"], type="ffd"),
                )

                self.connect(
                    MPhysVariables.Aerodynamics.Surface.Mesh.COORDINATES,
                    "geometry." + MPhysVariables.Aerodynamics.Surface.Geometry.COORDINATES_INPUT,
                )
                self.connect(
                    "geometry." + MPhysVariables.Aerodynamics.Surface.Geometry.COORDINATES_OUTPUT,
                    MPhysVariables.Aerodynamics.Surface.COORDINATES,
                )
                self.connect(
                    "geometry." + MPhysVariables.Aerodynamics.Surface.Geometry.COORDINATES_OUTPUT,
                    MPhysVariables.Aerodynamics.Surface.COORDINATES_INITIAL,
                )
            else:
                self.mphys_add_subsystem(
                    "mesh", aero_builder.get_mesh_coordinate_subsystem(self.name)
                )
                self.connect(
                    MPhysVariables.Aerodynamics.Surface.Mesh.COORDINATES,
                    MPhysVariables.Aerodynamics.Surface.COORDINATES,
                )
                self.connect(
                    MPhysVariables.Aerodynamics.Surface.Mesh.COORDINATES,
                    MPhysVariables.Aerodynamics.Surface.COORDINATES_INITIAL,
                )

        self._mphys_add_pre_coupling_subsystem_from_builder(
            "aero", aero_builder, self.name
        )
        self.mphys_add_subsystem(
            "coupling", aero_builder.get_coupling_group_subsystem(self.name)
        )
        self._mphys_add_post_coupling_subsystem_from_builder(
            "aero", aero_builder, self.name
        )
