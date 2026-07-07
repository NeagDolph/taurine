import './assets/main.css'

import { StrictMode, useEffect, useState } from 'react'
import { createRoot } from 'react-dom/client'
import App from './App'
import { PreviewView, previewNoteIdFromHash } from './views/PreviewView'

function Root(): React.JSX.Element {
  const [hash, setHash] = useState(window.location.hash)

  useEffect(() => {
    const onHashChange = (): void => setHash(window.location.hash)
    window.addEventListener('hashchange', onHashChange)
    return () => window.removeEventListener('hashchange', onHashChange)
  }, [])

  const noteId = previewNoteIdFromHash(hash)
  return noteId ? <PreviewView noteId={noteId} /> : <App />
}

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <Root />
  </StrictMode>
)
