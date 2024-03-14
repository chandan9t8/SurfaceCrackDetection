import java.rmi.Naming;
import common.Account;

public class AccountClient {
    public static void main(String[] args) {
        try {
            String remoteHostName = "in-csci-rrpc01.cs.iupui.edu";
            int port = 1099;
            String lookupString = "rmi://" + remoteHostName + ":" + port + "/AccountService";
            Account account = (Account) Naming.lookup(lookupString);
            
            // Example usage
            account.deposit(new Money(100));
            System.out.println("Balance after deposit: " + account.getBalance().getAmount());
            account.withdraw(new Money(50));
            System.out.println("Balance after withdrawal: " + account.getBalance().getAmount());
        } catch (Exception e) {
            System.err.println("Client exception: " + e.toString());
            e.printStackTrace();
        }
    }
}
