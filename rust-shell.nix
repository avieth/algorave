with import <nixpkgs> {};

# pkgs.rustPackages.

pkgs.rustPlatform.buildRustPackage {
  name = "hello-rust";
  version = "0.0.1";
  src = ./.;
  checkPhase = "";
  cargoSha256 = "sha256:1lym5h01if6mbiz1856ks1gj9gwfx9pj7a9jk6viq0ihrbavs9gk";
  buildInputs = [ jack2 ];
}
