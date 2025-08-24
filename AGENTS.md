# Agent Development Guidelines

## Build/Lint/Test Commands

### Frontend (agent-frontend/)
- **Dev server**: `bunx --bun vite` or `bun run dev`
- **Build**: `tsc -b && vite build` or `bun run build`
- **Lint**: `eslint .` or `bun run lint`
- **Preview**: `vite preview` or `bun run preview`

### Backend (agent-backend/)
- **Dev server**: `fastapi dev main.py`
- **Production**: `fastapi run main.py`
- **Install deps**: `uv sync` or `uv install`

## Code Style Guidelines

### TypeScript/React (Frontend)
- Use React 19 with modern hooks patterns
- Import statements: destructured imports for React components
- Strict TypeScript with `noUnusedLocals` and `noUnusedParameters`
- Use `react-jsx` transform, no explicit React imports needed
- File extensions: `.tsx` for components, `.ts` for utilities

### Python (Backend)
- Use type hints extensively (FastAPI + Pydantic style)
- Import grouping: stdlib, third-party, local imports with blank lines between
- Docstrings: detailed function/class documentation with examples
- Error handling: use FastAPI HTTPException for API errors
- Environment: Python 3.12+, FastAPI with async/await patterns
- Configuration: Pydantic Settings with .env support