import {
  Database,
  Upload,
  Network,
  MessageSquare,
  FileText,
  Download,
  Settings,
  Bot,
} from "lucide-react";
import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarHeader,
  SidebarFooter,
} from "@/components/ui/sidebar";
import { useLocation } from "wouter";

const menuItems = [
  {
    title: "Upload Documents",
    url: "/upload",
    icon: Upload,
  },
  {
    title: "Knowledge Base",
    url: "/knowledge-base",
    icon: Database,
  },
  {
    title: "Knowledge Graph",
    url: "/graph",
    icon: Network,
  },
  {
    title: "Research Assistant",
    url: "/chat",
    icon: MessageSquare,
  },
  {
    title: "Documents",
    url: "/documents",
    icon: FileText,
  },
];

const managementItems = [
  {
    title: "Statistics",
    url: "/statistics",
    icon: Database,
  },
  {
    title: "Agent Architecture",
    url: "/agents",
    icon: Bot,
  },
  {
    title: "Import/Export",
    url: "/import-export",
    icon: Download,
  },
];

export function AppSidebar() {
  const [location] = useLocation();

  return (
    <Sidebar>
      <SidebarHeader className="p-6">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-md bg-primary text-primary-foreground overflow-hidden">
            <div className="h-full w-full flex items-center justify-center bg-gradient-to-br from-blue-600 to-purple-600 text-white font-bold text-lg">
              NS
            </div>
          </div>
          <div>
            <h2 className="text-lg font-semibold">NeuroSymbolic</h2>
            <p className="text-xs text-muted-foreground">Human Resource Management</p>
          </div>
        </div>
      </SidebarHeader>
      <SidebarContent>
        <SidebarGroup>
          <SidebarGroupLabel>Main Functions</SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              {menuItems.map((item) => (
                <SidebarMenuItem key={item.title}>
                  <SidebarMenuButton asChild isActive={location === item.url}>
                    <a href={item.url} data-testid={`link-${item.url.slice(1)}`}>
                      <item.icon className="h-4 w-4" />
                      <span>{item.title}</span>
                    </a>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              ))}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
        <SidebarGroup>
          <SidebarGroupLabel>Management</SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              {managementItems.map((item) => (
                <SidebarMenuItem key={item.title}>
                  <SidebarMenuButton asChild isActive={location === item.url}>
                    <a href={item.url} data-testid={`link-${item.url.slice(1)}`}>
                      <item.icon className="h-4 w-4" />
                      <span>{item.title}</span>
                    </a>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              ))}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>
      <SidebarFooter className="p-4">
        <SidebarMenu>
          <SidebarMenuItem>
            <SidebarMenuButton asChild>
              <a href="/settings" data-testid="link-settings">
                <Settings className="h-4 w-4" />
                <span>Settings</span>
              </a>
            </SidebarMenuButton>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarFooter>
    </Sidebar>
  );
}
