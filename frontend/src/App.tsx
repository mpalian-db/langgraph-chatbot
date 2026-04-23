import { NavLink, Route, Routes } from "react-router-dom";
import ChatView from "./components/ChatView";
import CollectionsView from "./components/CollectionsView";

function App() {
  return (
    <div className="flex h-screen bg-gray-900 text-gray-100">
      <nav className="flex w-48 flex-col border-r border-gray-700 bg-gray-800">
        <div className="border-b border-gray-700 px-4 py-4">
          <span className="text-sm font-semibold text-white">
            LangGraph RAG
          </span>
        </div>

        <div className="flex flex-col gap-1 p-2">
          <NavLink
            to="/"
            end
            className={({ isActive }) =>
              `rounded px-3 py-2 text-sm font-medium transition-colors ${
                isActive
                  ? "bg-indigo-600/20 text-indigo-400"
                  : "text-gray-400 hover:bg-gray-700 hover:text-gray-200"
              }`
            }
          >
            Chat
          </NavLink>
          <NavLink
            to="/collections"
            className={({ isActive }) =>
              `rounded px-3 py-2 text-sm font-medium transition-colors ${
                isActive
                  ? "bg-indigo-600/20 text-indigo-400"
                  : "text-gray-400 hover:bg-gray-700 hover:text-gray-200"
              }`
            }
          >
            Collections
          </NavLink>
        </div>
      </nav>

      <main className="flex-1 overflow-hidden">
        <Routes>
          <Route path="/" element={<ChatView />} />
          <Route path="/collections" element={<CollectionsView />} />
        </Routes>
      </main>
    </div>
  );
}

export default App;
