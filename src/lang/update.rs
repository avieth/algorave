use crate::lang::instruction;

#[derive(Debug)]
pub struct MemoryUpdate {
    pub offset: u32,
    pub data: Vec<u8>
}

/// Updates to the system: you can set the program and/or update memory
/// locations.
#[derive(Debug)]
pub struct Update {
    pub program: Option<instruction::Program>,
    pub memory: Vec<MemoryUpdate>
}

impl Update {
    pub fn get_parts(self) -> (Option<instruction::Program>, Vec<MemoryUpdate>) {
        return (self.program, self.memory);
    }
}
