with import <nixpkgs> {};

# pkgs.rustPackages.

pkgs.rustPlatform.buildRustPackage {
  name = "hello-rust";
  version = "0.0.1";
  src = ./.;
  checkPhase = "";
  cargoSha256 = "sha256:19x7dqigs5c56q9xncqrs7anz9z0as390nvc73gasj5rwn0xw6pw";
  buildInputs = [ jack2 ];
}
